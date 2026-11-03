import csv
import random
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import SimpleCNN, IMAGE_SIZE, NUM_CLASSES


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = "color"

PARTITION_FILE = Path(
    "mixed_10iid_10noniid_partitions.npz"
)

NUM_CLIENTS = 20
NUM_IID_CLIENTS = 10
NUM_NONIID_CLIENTS = 10

NUM_ROUNDS = 100

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SEED = 42

# Exactly TWO Non-IID clients removed every round
NUM_NONIID_TO_REMOVE = 2


# ============================================================
# 2. OUTPUT FILES
# ============================================================

ACCURACY_CSV = Path(
    "karthi_100round_accuracy.csv"
)

ENTROPY_CSV = Path(
    "karthi_100round_entropy.csv"
)

FINAL_MODEL = Path(
    "karthi_100round_final_model.pt"
)


# ============================================================
# 3. RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nUsing device: {DEVICE}",
    flush=True,
)


# ============================================================
# 5. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


# ============================================================
# 6. LOAD DATASET
# ============================================================

full_dataset = ImageFolder(
    DATA_DIR,
    transform=transform,
)

if len(full_dataset) == 0:
    raise RuntimeError(
        "Dataset is empty."
    )

if len(full_dataset.classes) != NUM_CLASSES:
    raise RuntimeError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(full_dataset.classes)}."
    )

print(
    f"Total images: {len(full_dataset)}",
    flush=True,
)

print(
    f"Total classes: {len(full_dataset.classes)}",
    flush=True,
)


# ============================================================
# 7. LOAD SAME SAVED PARTITION
# ============================================================

if not PARTITION_FILE.exists():

    raise FileNotFoundError(
        f"Could not find {PARTITION_FILE}"
    )

partitions = np.load(
    PARTITION_FILE,
    allow_pickle=True,
)

saved_dataset_size = int(
    partitions["dataset_size"]
)

if saved_dataset_size != len(full_dataset):

    raise RuntimeError(
        "Dataset size does not match "
        "the saved partition."
    )


# ============================================================
# 8. BUILD CLIENT DATASETS
# ============================================================

client_train_datasets = []
client_val_datasets = []

for cid in range(NUM_CLIENTS):

    train_indices = partitions[
        f"train_{cid}"
    ]

    val_indices = partitions[
        f"val_{cid}"
    ]

    client_train_datasets.append(
        Subset(
            full_dataset,
            train_indices,
        )
    )

    client_val_datasets.append(
        Subset(
            full_dataset,
            val_indices,
        )
    )


# ============================================================
# 9. GLOBAL TEST DATA
# ============================================================

global_test_indices = partitions[
    "global_test_indices"
]

global_test_dataset = Subset(
    full_dataset,
    global_test_indices,
)

global_test_loader = DataLoader(
    global_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(
    f"Global test images: "
    f"{len(global_test_dataset)}",
    flush=True,
)


# ============================================================
# 10. SHOW CLIENT INFORMATION
# ============================================================

print(
    "\n========================================"
)

print(
    " CLIENT DATA"
)

print(
    "========================================"
)

for cid in range(NUM_CLIENTS):

    client_type = (
        "IID"
        if cid < NUM_IID_CLIENTS
        else "Non-IID"
    )

    print(
        f"Client {cid:02d} | "
        f"{client_type:7s} | "
        f"train={len(client_train_datasets[cid])} | "
        f"val={len(client_val_datasets[cid])}"
    )


# ============================================================
# 11. MODEL PARAMETER FUNCTIONS
# ============================================================

def get_model_parameters(model):

    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_model_parameters(
    model,
    parameters,
):

    state_dict = model.state_dict()

    new_state_dict = {}

    for key, value in zip(
        state_dict.keys(),
        parameters,
    ):

        new_state_dict[key] = torch.tensor(
            value
        )

    model.load_state_dict(
        new_state_dict,
        strict=True,
    )


# ============================================================
# 12. LOCAL TRAINING
# ============================================================

def train_locally(
    model,
    dataloader,
):

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_loss = 0.0
    total_samples = 0

    for _ in range(LOCAL_EPOCHS):

        for images, targets in dataloader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return (
        total_loss
        / total_samples
    )


# ============================================================
# 13. SHANNON PREDICTION ENTROPY
#
# H = - sum(p * log(p))
#
# Measured AFTER local training
# on the client's private validation set.
# ============================================================

def compute_prediction_entropy(
    model,
    dataloader,
):

    model.eval()

    entropy_values = []

    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = F.softmax(
                outputs,
                dim=1,
            )

            entropy = -torch.sum(
                probabilities
                * torch.log(
                    probabilities + 1e-12
                ),
                dim=1,
            )

            entropy_values.extend(
                entropy
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    if len(entropy_values) == 0:
        return 0.0

    return float(
        np.mean(entropy_values)
    )


# ============================================================
# 14. GLOBAL TEST ACCURACY
# ============================================================

def evaluate_accuracy(
    model,
    dataloader,
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

    if total == 0:
        return 0.0

    return correct / total


# ============================================================
# 15. FLOWER CLIENT
# ============================================================

class FlowerClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id,
    ):

        self.client_id = client_id

        self.model = SimpleCNN(
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        self.train_loader = DataLoader(
            client_train_datasets[
                client_id
            ],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            client_val_datasets[
                client_id
            ],
            batch_size=BATCH_SIZE,
            shuffle=False,
        )


    def get_parameters(
        self,
        config,
    ):

        return get_model_parameters(
            self.model
        )


    def fit(
        self,
        parameters,
        config,
    ):

        # Receive current global model
        set_model_parameters(
            self.model,
            parameters,
        )

        # Local training
        local_loss = train_locally(
            self.model,
            self.train_loader,
        )

        # Shannon prediction entropy
        # AFTER local training
        entropy = compute_prediction_entropy(
            self.model,
            self.val_loader,
        )

        client_type = (
            "IID"
            if self.client_id < NUM_IID_CLIENTS
            else "Non-IID"
        )

        print(
            f"Client {self.client_id:02d} | "
            f"{client_type:7s} | "
            f"loss={local_loss:.4f} | "
            f"H={entropy:.6f}",
            flush=True,
        )

        metrics = {

            "client_id":
                int(self.client_id),

            "entropy":
                float(entropy),

            "client_type":
                client_type,

            "local_loss":
                float(local_loss),
        }

        return (

            get_model_parameters(
                self.model
            ),

            len(
                self.train_loader.dataset
            ),

            metrics,
        )


    def evaluate(
        self,
        parameters,
        config,
    ):

        set_model_parameters(
            self.model,
            parameters,
        )

        return (
            0.0,
            len(
                self.val_loader.dataset
            ),
            {},
        )


# ============================================================
# 16. CLIENT FUNCTION
# ============================================================

def client_fn(cid):

    return FlowerClient(
        int(cid)
    ).to_client()


# ============================================================
# 17. CREATE CSV FILES
# ============================================================

with open(
    ACCURACY_CSV,
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "round",
        "accuracy",
        "kept_clients",
        "removed_clients",
    ])


with open(
    ENTROPY_CSV,
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "round",
        "client_id",
        "client_type",
        "entropy",
        "status",
    ])


# ============================================================
# 18. HISTORY
# ============================================================

accuracy_history = []

final_parameters = None


# ============================================================
# 19. KARTHI FEDAVG
# ============================================================

class KarthiFedAvg(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        global final_parameters

        print(
            "\n\n"
            "========================================"
        )

        print(
            f" ROUND {server_round} "
            f"- KARTHI ENTROPY SELECTION"
        )

        print(
            "========================================"
        )


        # ----------------------------------------------------
        # STEP 1:
        # Separate IID and Non-IID clients
        # ----------------------------------------------------

        iid_records = []
        noniid_records = []

        for client_proxy, fit_res in results:

            metrics = fit_res.metrics

            cid = int(
                metrics["client_id"]
            )

            entropy = float(
                metrics["entropy"]
            )

            record = {

                "cid":
                    cid,

                "entropy":
                    entropy,

                "client_proxy":
                    client_proxy,

                "fit_res":
                    fit_res,
            }

            if cid < NUM_IID_CLIENTS:

                iid_records.append(
                    record
                )

            else:

                noniid_records.append(
                    record
                )


        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        if len(iid_records) != 10:

            raise RuntimeError(
                f"Expected 10 IID results, "
                f"received {len(iid_records)}."
            )

        if len(noniid_records) != 10:

            raise RuntimeError(
                f"Expected 10 Non-IID results, "
                f"received {len(noniid_records)}."
            )


        # ----------------------------------------------------
        # STEP 2:
        # SORT ONLY NON-IID CLIENTS
        #
        # LOWEST entropy first
        # ----------------------------------------------------

        noniid_records.sort(
            key=lambda x: x["entropy"]
        )


        # ----------------------------------------------------
        # STEP 3:
        # REMOVE EXACTLY TWO LOWEST
        # ----------------------------------------------------

        removed_noniid = (
            noniid_records[
                :NUM_NONIID_TO_REMOVE
            ]
        )

        kept_noniid = (
            noniid_records[
                NUM_NONIID_TO_REMOVE:
            ]
        )


        removed_ids = [
            record["cid"]
            for record in removed_noniid
        ]

        kept_noniid_ids = [
            record["cid"]
            for record in kept_noniid
        ]


        # ----------------------------------------------------
        # ALL IID CLIENTS ARE ALWAYS KEPT
        # ----------------------------------------------------

        iid_records.sort(
            key=lambda x: x["cid"]
        )

        iid_ids = [
            record["cid"]
            for record in iid_records
        ]


        # ----------------------------------------------------
        # COMBINE:
        #
        # 10 IID + 8 Non-IID = 18
        # ----------------------------------------------------

        kept_records = (
            iid_records
            + kept_noniid
        )

        kept_results = [

            (
                record["client_proxy"],
                record["fit_res"],
            )

            for record
            in kept_records
        ]


        # ----------------------------------------------------
        # PRINT NON-IID RANKING
        # ----------------------------------------------------

        print(
            "\nNon-IID entropy ranking "
            "(LOWEST to HIGHEST):\n"
        )

        for rank, record in enumerate(
            noniid_records,
            start=1,
        ):

            cid = record["cid"]
            entropy = record["entropy"]

            if cid in removed_ids:
                status = "REMOVE"
            else:
                status = "KEEP"

            print(
                f"Rank {rank:02d} | "
                f"Client {cid:02d} | "
                f"H={entropy:.6f} | "
                f"{status}"
            )


        # ----------------------------------------------------
        # SAVE ALL CLIENT DECISIONS
        # ----------------------------------------------------

        all_records = (
            iid_records
            + noniid_records
        )

        all_records.sort(
            key=lambda x: x["cid"]
        )

        for record in all_records:

            cid = record["cid"]

            entropy = record["entropy"]

            client_type = (
                "IID"
                if cid < NUM_IID_CLIENTS
                else "Non-IID"
            )

            if cid < NUM_IID_CLIENTS:

                status = "KEEP"

            elif cid in removed_ids:

                status = "REMOVE"

            else:

                status = "KEEP"


            with open(
                ENTROPY_CSV,
                "a",
                newline="",
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow([
                    server_round,
                    cid,
                    client_type,
                    entropy,
                    status,
                ])


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print(
            "\n----------------------------------------"
        )

        print(
            f"IID clients kept: "
            f"{iid_ids}"
        )

        print(
            f"Non-IID clients kept: "
            f"{kept_noniid_ids}"
        )

        print(
            f"Non-IID clients removed: "
            f"{removed_ids}"
        )

        print(
            f"Total updates for FedAvg: "
            f"{len(kept_results)}"
        )

        print(
            "----------------------------------------"
        )


        # Critical safety check
        if len(kept_results) != 18:

            raise RuntimeError(
                f"Karthi method should aggregate "
                f"exactly 18 updates, but got "
                f"{len(kept_results)}."
            )


        # ----------------------------------------------------
        # STEP 4:
        # FEDAVG THE 18 KEPT UPDATES
        # ----------------------------------------------------

        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                kept_results,
                failures,
            )
        )

        if aggregated_parameters is None:

            print(
                "Aggregation failed."
            )

            return (
                aggregated_parameters,
                metrics,
            )


        final_parameters = (
            aggregated_parameters
        )


        # ----------------------------------------------------
        # STEP 5:
        # GLOBAL MODEL ACCURACY
        # ----------------------------------------------------

        global_model = SimpleCNN(
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        set_model_parameters(
            global_model,
            parameters_to_ndarrays(
                aggregated_parameters
            ),
        )

        accuracy = evaluate_accuracy(
            global_model,
            global_test_loader,
        )

        accuracy_history.append(
            (
                server_round,
                accuracy,
            )
        )


        # ----------------------------------------------------
        # SAVE ACCURACY
        # ----------------------------------------------------

        with open(
            ACCURACY_CSV,
            "a",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([
                server_round,
                accuracy,
                len(kept_results),
                len(removed_ids),
            ])


        # ----------------------------------------------------
        # PRINT ROUND RESULT
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            f"Round {server_round:02d}/"
            f"{NUM_ROUNDS}"
        )

        print(
            f"Removed Non-IID clients = "
            f"{removed_ids}"
        )

        print(
            f"Aggregated updates = "
            f"{len(kept_results)}"
        )

        print(
            f"Global test accuracy = "
            f"{accuracy:.4f} "
            f"({accuracy * 100:.2f}%)"
        )

        print(
            "========================================\n"
        )

        return (
            aggregated_parameters,
            metrics,
        )


# ============================================================
# 20. START FEDERATED LEARNING
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        " KARTHI IDEA - 5 ROUND TEST"
    )

    print(
        "========================================"
    )

    print(
        "\n20 clients:"
    )

    print(
        "Clients 0-9   = IID "
        "(always kept)"
    )

    print(
        "Clients 10-19 = Non-IID "
        "(entropy ranked)"
    )

    print(
        "\nEntropy:"
    )

    print(
        "Shannon prediction entropy "
        "AFTER local training"
    )

    print(
        "\nSelection:"
    )

    print(
        "Remove exactly 2 LOWEST-entropy "
        "Non-IID clients"
    )

    print(
        "Keep 10 IID + 8 Non-IID"
    )

    print(
        "FedAvg = 18 updates per round"
    )

    print(
        f"\nRounds: {NUM_ROUNDS}"
    )

    print(
        f"Local epochs: {LOCAL_EPOCHS}\n"
    )


    # --------------------------------------------------------
    # INITIAL GLOBAL MODEL
    # --------------------------------------------------------

    initial_model = SimpleCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    initial_parameters = (
        ndarrays_to_parameters(
            get_model_parameters(
                initial_model
            )
        )
    )


    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    strategy = KarthiFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=0.0,

        min_fit_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        min_evaluate_clients=0,

        initial_parameters=(
            initial_parameters
        ),
    )


    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    history = fl.simulation.start_simulation(

        client_fn=client_fn,

        num_clients=NUM_CLIENTS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": 1,

            "num_gpus": (
                0.2
                if torch.cuda.is_available()
                else 0.0
            ),
        },
    )


    # ========================================================
    # 21. SAVE FINAL MODEL
    # ========================================================

    if final_parameters is not None:

        final_model = SimpleCNN(
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        set_model_parameters(
            final_model,
            parameters_to_ndarrays(
                final_parameters
            ),
        )

        torch.save(
            final_model.state_dict(),
            FINAL_MODEL,
        )

        print(
            f"\nFinal model saved to: "
            f"{FINAL_MODEL}"
        )


    # ========================================================
    # 22. FINAL SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " KARTHI 5-ROUND TEST FINISHED"
    )

    print(
        "========================================"
    )

    for round_number, accuracy in (
        accuracy_history
    ):

        print(
            f"Round {round_number:02d}: "
            f"{accuracy * 100:.2f}%"
        )

    print(
        f"\nAccuracy CSV: "
        f"{ACCURACY_CSV}"
    )

    print(
        f"Entropy CSV: "
        f"{ENTROPY_CSV}"
    )

    print(
        "\nDone."
    )
import matplotlib.pyplot as plt

# =========================
# PLOT ACCURACY GRAPH
# =========================

rounds = [row[0] for row in accuracy_history]
accuracies = [row[1] * 100 for row in accuracy_history]

plt.figure(figsize=(10, 6))

plt.plot(
    rounds,
    accuracies,
    marker="o",
    markersize=3,
    linewidth=2
)

plt.xlabel("Communication Round")
plt.ylabel("Global Test Accuracy (%)")
plt.title("Karthi Idea - Global Accuracy over 100 Rounds")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "karthi_100round_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Accuracy graph saved to: karthi_100round_accuracy.png")
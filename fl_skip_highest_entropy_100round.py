# Experiment: remove 2 highest-prediction-entropy Non-IID updates per round.
# All 20 clients train each round; 10 IID + 8 Non-IID are aggregated.
# Derived from the existing Flower implementation; no class counts shared.
# ============================================================
# SKIP HIGHEST NON-IID - 100 ROUNDS
#
# 20 clients:
#   Clients 0-9  = IID
#   Clients 10-19 = Non-IID
#
# Every round:
#   1. All 20 clients receive the global model
#   2. All 20 train for 1 local epoch
#   3. Calculate prediction entropy AFTER training
#   4. Server calculates:
#
#          delta = mean entropy of all 20 clients
#
#   5. H >= delta  -> KEEP
#      H <  delta  -> REMOVE
#
#   6. FedAvg only the KEPT client updates
#   7. Evaluate global model
# ============================================================


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

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import (
    SimpleCNN,
    IMAGE_SIZE,
    NUM_CLASSES,
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = "color"

PARTITION_FILE = Path(
    "mixed_10iid_10noniid_partitions.npz"
)

NUM_CLIENTS = 20
NUM_IID_CLIENTS = 10

# FIRST TEST ONLY
NUM_ROUNDS = 100

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SEED = 42


# Output files
ACCURACY_CSV = Path(
    "skip_highest_100round_accuracy.csv"
)

ENTROPY_CSV = Path(
    "skip_highest_100round_entropy.csv"
)

FINAL_MODEL = Path(
    "skip_highest_100round_final_model.pt"
)


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE
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
# 4. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
])


# ============================================================
# 5. LOAD FULL DATASET
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
        f"but found "
        f"{len(full_dataset.classes)}."
    )


print(
    f"Total images: {len(full_dataset)}",
    flush=True,
)

print(
    f"Total classes: "
    f"{len(full_dataset.classes)}",
    flush=True,
)


# ============================================================
# 6. LOAD SAVED 20-CLIENT PARTITION
# ============================================================

if not PARTITION_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{PARTITION_FILE}\n\n"
        f"Make sure the saved partition "
        f"file is in this folder."
    )


partitions = np.load(
    PARTITION_FILE,
    allow_pickle=True,
)


# Check dataset size
saved_dataset_size = int(
    partitions["dataset_size"]
)

if saved_dataset_size != len(full_dataset):

    raise RuntimeError(
        "Dataset size does not match "
        "the saved partition."
    )


# ============================================================
# 7. BUILD CLIENT DATASETS
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
# 8. GLOBAL TEST DATA
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
# 9. SHOW CLIENT INFORMATION
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
        f"train="
        f"{len(client_train_datasets[cid])} | "
        f"val="
        f"{len(client_val_datasets[cid])}"
    )


# ============================================================
# 10. MODEL PARAMETER FUNCTIONS
# ============================================================

def get_model_parameters(model):

    return [
        value.detach().cpu().numpy()
        for value
        in model.state_dict().values()
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

        new_state_dict[key] = (
            torch.tensor(value)
        )

    model.load_state_dict(
        new_state_dict,
        strict=True,
    )


# ============================================================
# 11. LOCAL TRAINING
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
# 12. PREDICTION ENTROPY
#
# H = - sum(p * log(p))
#
# IMPORTANT:
# Entropy is measured AFTER local training
# using the client's private validation data.
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
# 13. GLOBAL TEST ACCURACY
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
# 14. FLOWER CLIENT
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


    # --------------------------------------------------------
    # GET PARAMETERS
    # --------------------------------------------------------

    def get_parameters(
        self,
        config,
    ):

        return get_model_parameters(
            self.model
        )


    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    def fit(
        self,
        parameters,
        config,
    ):

        # Every round the client receives
        # the SAME current global model.
        set_model_parameters(
            self.model,
            parameters,
        )


        # Train locally
        local_loss = train_locally(
            self.model,
            self.train_loader,
        )


        # Prediction entropy AFTER training
        h_end = compute_prediction_entropy(
            self.model,
            self.val_loader,
        )


        client_type = (
            "IID"
            if self.client_id
            < NUM_IID_CLIENTS
            else "Non-IID"
        )


        print(
            f"Client "
            f"{self.client_id:02d} | "
            f"{client_type:7s} | "
            f"loss={local_loss:.4f} | "
            f"H={h_end:.6f}",
            flush=True,
        )


        # IMPORTANT:
        #
        # The client DOES NOT decide whether
        # it is accepted or rejected.
        #
        # It sends:
        #
        # 1. Model update
        # 2. Entropy value
        #
        # Server will calculate delta after
        # receiving ALL 20 clients.
        metrics = {

            "client_id":
                int(self.client_id),

            "entropy":
                float(h_end),

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


    # --------------------------------------------------------
    # CLIENT EVALUATION
    # --------------------------------------------------------

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
# 15. CLIENT FUNCTION
# ============================================================

def client_fn(cid):

    return FlowerClient(
        int(cid)
    ).to_client()


# ============================================================
# 16. CSV FILES
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
        "second_highest_noniid_entropy",
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
        "second_highest_noniid_entropy",
        "status",
    ])


# ============================================================
# 17. HISTORY
# ============================================================

accuracy_history = []

final_parameters = None


# ============================================================
# 18. FARZANA FEDAVG
# ============================================================

class SkipHighestFedAvg(
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
            f"- SKIP HIGHEST NON-IID ENTROPY"
        )

        print(
            "========================================"
        )


        # ----------------------------------------------------
        # STEP 1:
        # Collect entropy from ALL clients
        # ----------------------------------------------------

        entropy_records = []


        for client_proxy, fit_res in results:

            metrics = fit_res.metrics

            cid = int(
                metrics["client_id"]
            )

            entropy = float(
                metrics["entropy"]
            )

            client_type = str(
                metrics["client_type"]
            )


            entropy_records.append({

                "cid":
                    cid,

                "entropy":
                    entropy,

                "client_type":
                    client_type,

                "client_proxy":
                    client_proxy,

                "fit_res":
                    fit_res,
            })


        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if len(entropy_records) == 0:

            print(
                "No client results received."
            )

            return None, {}


        # ----------------------------------------------------
        # STEP 2: Rank only Non-IID clients (IDs 10-19).
        # Remove the TWO highest Shannon prediction entropy values.
        # IID clients are never excluded by this rule.
        # Entropy is recalculated locally after training each round.
        # ----------------------------------------------------
        noniid_records = [r for r in entropy_records if r["cid"] >= NUM_IID_CLIENTS]
        if len(noniid_records) < 3:
            raise RuntimeError("Expected at least three Non-IID updates to remove two safely")
        ranked_noniid = sorted(noniid_records, key=lambda r: (-r["entropy"], r["cid"]))
        removed_ids = {r["cid"] for r in ranked_noniid[:2]}
        delta = float(ranked_noniid[1]["entropy"])  # logging only: second-highest Non-IID entropy
        kept_results = []
        kept_client_ids = []
        removed_client_ids = []
        for record in sorted(entropy_records, key=lambda r: r["cid"]):
            cid = record["cid"]
            entropy = record["entropy"]
            client_type = record["client_type"]
            if cid in removed_ids:
                status = "REMOVE"
                removed_client_ids.append(cid)
            else:
                status = "KEEP"
                kept_results.append((record["client_proxy"], record["fit_res"]))
                kept_client_ids.append(cid)
            print(f"Client {cid:02d} | {client_type:7s} | H={entropy:.6f} | {status}")
            with open(ENTROPY_CSV, "a", newline="") as file:
                csv.writer(file).writerow([server_round, cid, client_type, entropy, delta, status])
        print(f"Removed two highest Non-IID entropy clients: {sorted(removed_ids)}")

        # ----------------------------------------------------
        # ----------------------------------------------------
        # SHOW SUMMARY
        # ----------------------------------------------------

        print(
            "\n----------------------------------------"
        )

        print(
            f"Total clients: "
            f"{len(entropy_records)}"
        )

        print(
            f"Kept: "
            f"{len(kept_client_ids)}"
        )

        print(
            f"Removed: "
            f"{len(removed_client_ids)}"
        )

        print(
            f"Kept client IDs: "
            f"{kept_client_ids}"
        )

        print(
            f"Removed client IDs: "
            f"{removed_client_ids}"
        )

        print(
            "----------------------------------------"
        )


        # ----------------------------------------------------
        # SAFETY:
        # Need at least one update
        # ----------------------------------------------------

        if len(kept_results) == 0:

            print(
                "\nERROR: "
                "Entropy gate removed "
                "every client."
            )

            return None, {}


        # ----------------------------------------------------
        # STEP 4:
        # FEDAVG ONLY KEPT CLIENTS
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


        # Keep final parameters
        final_parameters = (
            aggregated_parameters
        )


        # ----------------------------------------------------
        # STEP 5:
        # EVALUATE NEW GLOBAL MODEL
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
                delta,
                len(
                    kept_client_ids
                ),
                len(
                    removed_client_ids
                ),
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
            f"Second-highest Non-IID entropy = "
            f"{delta:.6f}"
        )

        print(
            f"Aggregated updates = "
            f"{len(kept_client_ids)}"
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
# 19. START FEDERATED LEARNING
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        " SKIP HIGHEST NON-IID - 100 ROUNDS"
    )

    print(
        "========================================"
    )

    print(
        "\n20 clients:"
    )

    print(
        "Clients 0-9   = IID"
    )

    print(
        "Clients 10-19 = Non-IID"
    )

    print(
        "\nEntropy:"
    )

    print(
        "Prediction entropy AFTER "
        "local training"
    )

    print(
        "\nGate:"
    )

    print(
        "Remove the two highest-entropy Non-IID updates"
    )

    print(
        "Keep all 10 IID client updates"
    )

    print(
        "\nRounds: 100"
    )

    print(
        "Local epochs: 1\n"
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

    strategy = SkipHighestFedAvg(

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
    # RUN SIMULATION
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
    # 20. SAVE FINAL GLOBAL MODEL
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
    # 21. FINAL SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        " SKIP HIGHEST 100-ROUND TEST FINISHED"
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
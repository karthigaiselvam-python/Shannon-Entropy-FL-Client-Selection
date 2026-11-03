import random
from collections import Counter

import numpy as np
from torchvision.datasets import ImageFolder


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_DIR = "color"

NUM_CLIENTS = 20
NUM_IID_CLIENTS = 10
NUM_NONIID_CLIENTS = 10

ALPHA = 0.1
TRAIN_SPLIT = 0.90
CLIENT_VAL_SPLIT = 0.10

SEED = 42
PARTITION_FILE = "mixed_10iid_10noniid_partitions.npz"

# We need at least train + validation images for every client.
MIN_CLIENT_SAMPLES = 2


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

rng = np.random.default_rng(SEED)


# ============================================================
# 3. LOAD PLANTVILLAGE
# ============================================================

dataset = ImageFolder(DATA_DIR)

labels = np.asarray(dataset.targets)

print(f"\nTotal images: {len(dataset)}")
print(f"Total classes: {len(dataset.classes)}")


# ============================================================
# 4. GLOBAL 90/10 TRAIN-TEST SPLIT
# ============================================================

all_indices = list(range(len(dataset)))

random.Random(SEED).shuffle(all_indices)

split_point = int(len(all_indices) * TRAIN_SPLIT)

client_pool_indices = np.asarray(
    all_indices[:split_point],
    dtype=np.int64,
)

global_test_indices = np.asarray(
    all_indices[split_point:],
    dtype=np.int64,
)

print(f"Client pool images: {len(client_pool_indices)}")
print(f"Global test images: {len(global_test_indices)}")


# ============================================================
# 5. DIVIDE CLIENT POOL INTO TWO HALVES
#
#    Half 1 -> 10 IID clients
#    Half 2 -> 10 Non-IID clients
#
# Important:
# Both halves come from the SAME global training pool.
# ============================================================

mixed_pool = client_pool_indices.copy()
rng.shuffle(mixed_pool)

half_point = len(mixed_pool) // 2

iid_pool = mixed_pool[:half_point]
noniid_pool = mixed_pool[half_point:]


print("\nData available for each group:")
print(f"IID pool:     {len(iid_pool)}")
print(f"Non-IID pool: {len(noniid_pool)}")


# ============================================================
# 6. CREATE 10 IID CLIENTS
# ============================================================

iid_pool = iid_pool.copy()
rng.shuffle(iid_pool)

iid_chunks = np.array_split(
    iid_pool,
    NUM_IID_CLIENTS,
)


# ============================================================
# 7. CREATE 10 NON-IID CLIENTS USING DIRICHLET alpha = 0.1
# ============================================================

def create_noniid_clients(pool_indices):

    for attempt in range(1000):

        client_indices = [
            [] for _ in range(NUM_NONIID_CLIENTS)
        ]

        pool_labels = labels[pool_indices]

        for class_id in np.unique(pool_labels):

            class_indices = pool_indices[
                pool_labels == class_id
            ].copy()

            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(
                    NUM_NONIID_CLIENTS,
                    ALPHA,
                )
            )

            cut_points = (
                np.cumsum(proportions)[:-1]
                * len(class_indices)
            ).astype(int)

            chunks = np.split(
                class_indices,
                cut_points,
            )

            for local_cid, chunk in enumerate(chunks):

                client_indices[local_cid].extend(
                    chunk.tolist()
                )

        client_sizes = [
            len(indices)
            for indices in client_indices
        ]

        if min(client_sizes) >= MIN_CLIENT_SAMPLES:
            return client_indices

    raise RuntimeError(
        "Could not create valid Non-IID partitions."
    )


noniid_chunks = create_noniid_clients(
    noniid_pool
)


# ============================================================
# 8. COMBINE INTO 20 CLIENTS
#
# Clients 0-9   = IID
# Clients 10-19 = Non-IID
# ============================================================

client_indices = []

for chunk in iid_chunks:
    client_indices.append(
        list(map(int, chunk))
    )

for chunk in noniid_chunks:
    client_indices.append(
        list(map(int, chunk))
    )


# ============================================================
# 9. CREATE PRIVATE TRAIN / VALIDATION SPLIT FOR EACH CLIENT
# ============================================================

client_train_indices = []
client_val_indices = []

for cid, indices in enumerate(client_indices):

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    rng.shuffle(indices)

    val_size = max(
        1,
        int(len(indices) * CLIENT_VAL_SPLIT),
    )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    client_train_indices.append(
        train_indices.tolist()
    )

    client_val_indices.append(
        val_indices.tolist()
    )


# ============================================================
# 10. SAFETY CHECKS
# ============================================================

all_client_indices = []

for cid in range(NUM_CLIENTS):

    if len(client_train_indices[cid]) == 0:
        raise RuntimeError(
            f"Client {cid} has no training images."
        )

    if len(client_val_indices[cid]) == 0:
        raise RuntimeError(
            f"Client {cid} has no validation images."
        )

    all_client_indices.extend(
        client_train_indices[cid]
    )

    all_client_indices.extend(
        client_val_indices[cid]
    )


if len(all_client_indices) != len(set(all_client_indices)):
    raise RuntimeError(
        "Some images appear in more than one client."
    )


if set(all_client_indices).intersection(
    global_test_indices.tolist()
):
    raise RuntimeError(
        "Client data overlaps with global test data."
    )


if (
    len(all_client_indices)
    + len(global_test_indices)
    != len(dataset)
):
    raise RuntimeError(
        "The partitions do not cover the complete dataset."
    )


print("\nPartition safety checks PASSED.")


# ============================================================
# 11. SHOW CLIENT DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("IID CLIENTS (0-9)")
print("=" * 70)

for cid in range(10):

    client_labels = labels[
        client_train_indices[cid]
    ]

    counts = Counter(
        client_labels.tolist()
    )

    print(
        f"Client {cid:02d} | "
        f"train={len(client_train_indices[cid]):5d} | "
        f"val={len(client_val_indices[cid]):4d} | "
        f"classes={len(counts):2d}"
    )

    print(
        "Top 5 classes:",
        counts.most_common(5),
    )


print("\n" + "=" * 70)
print("NON-IID CLIENTS (10-19) | Dirichlet alpha = 0.1")
print("=" * 70)

for cid in range(10, 20):

    client_labels = labels[
        client_train_indices[cid]
    ]

    counts = Counter(
        client_labels.tolist()
    )

    print(
        f"Client {cid:02d} | "
        f"train={len(client_train_indices[cid]):5d} | "
        f"val={len(client_val_indices[cid]):4d} | "
        f"classes={len(counts):2d}"
    )

    print(
        "Top 5 classes:",
        counts.most_common(5),
    )


print("\nMixed IID + Non-IID split created successfully.")
# ============================================================
# 12. SAVE THE EXACT 20-CLIENT PARTITION
# ============================================================

save_data = {
    "global_test_indices": np.asarray(
        global_test_indices,
        dtype=np.int64,
    ),
    "dataset_size": np.asarray(len(dataset)),
    "num_clients": np.asarray(NUM_CLIENTS),
    "num_iid_clients": np.asarray(NUM_IID_CLIENTS),
    "num_noniid_clients": np.asarray(NUM_NONIID_CLIENTS),
    "alpha": np.asarray(ALPHA),
    "seed": np.asarray(SEED),
    "class_names": np.asarray(dataset.classes),
}

for cid in range(NUM_CLIENTS):

    save_data[f"train_{cid}"] = np.asarray(
        client_train_indices[cid],
        dtype=np.int64,
    )

    save_data[f"val_{cid}"] = np.asarray(
        client_val_indices[cid],
        dtype=np.int64,
    )


np.savez_compressed(
    PARTITION_FILE,
    **save_data,
)

print(
    f"\nSaved exact 20-client partition to: {PARTITION_FILE}"
)
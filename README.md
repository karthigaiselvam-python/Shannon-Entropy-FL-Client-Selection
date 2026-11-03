# Shannon Entropy-Based Client Selection in Federated Learning

## Project Overview

This repository explores Shannon predictive entropy as a criterion for selecting clients in Federated Learning under heterogeneous (Non-IID) data distributions.

The project compares three entropy-based client selection strategies against a standard Federated Averaging (FedAvg) baseline, using agricultural leaf images from the PlantVillage dataset.

## Technologies Used

- Python
- PyTorch
- Flower (Federated Learning)
- Convolutional Neural Networks (CNN)
- NumPy
- PlantVillage Dataset

## Experimental Setup

- **Dataset:** PlantVillage
- **Classification task:** Plant disease classification across 38 classes
- **Clients:** 20 simulated clients
- **Data distribution:** 10 IID and 10 Non-IID clients
- **Non-IID generation:** Dirichlet distribution (α = 0.1)
- **Federated aggregation:** FedAvg
- **Communication rounds:** 100

## Client Selection Methods

### 1. FedAvg Baseline

All participating clients contribute their local model updates without entropy-based filtering.

### 2. Farzana Method — Mean Entropy Threshold

Shannon predictive entropy is calculated for each client. The mean entropy across clients is used as a dynamic selection threshold.

Clients with entropy greater than or equal to the threshold are selected for aggregation.

### 3. Karthi Method — Lowest-Two Exclusion

All IID clients are retained. The two Non-IID clients with the lowest predictive entropy are excluded from aggregation.

The remaining client updates are aggregated using FedAvg.

### 4. Highest-Entropy Exclusion

Clients are ranked according to their predictive entropy, and the two highest-entropy clients are excluded from aggregation.

## Repository Files

| File | Description |
|---|---|
| `model.py` | CNN model implementation |
| `mixed_data_split.py` | IID and Non-IID data preparation |
| `fl_fedavg_baseline_100round.py` | FedAvg baseline experiment |
| `fl_farzana_100round.py` | Mean entropy threshold experiment |
| `fl_karthi_100round.py` | Lowest-two exclusion experiment |
| `fl_skip_highest_entropy_100round.py` | Highest-entropy exclusion experiment |
| `requirements.txt` | Python dependencies |

## Research Objective

To investigate how different Shannon entropy-based client selection strategies influence federated model performance under heterogeneous agricultural image data distributions.

## Notes

The experiments are intended for research and comparative evaluation. Performance may vary with data partitioning, initialization, and training conditions.

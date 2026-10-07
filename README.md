# 🛡️ Privacy-Preserving Federated Symptom Checker

> A mobile-first diagnostic assistant that performs **on-device** symptom classification while preserving user privacy through **Federated Learning** and **Differential Privacy**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.5+-green.svg)](https://flower.ai)
[![Opacus](https://img.shields.io/badge/Opacus-1.4+-purple.svg)](https://opacus.ai)

---

## 🏗️ Architecture

The system consists of **three layers**:

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Client-Side (Local ML)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Symptom  │  │   Skin   │  │ Respira- │  ← On-device │
│  │   MLP    │  │   CNN    │  │ tory CNN │    models     │
│  └──────────┘  └──────────┘  └──────────┘              │
│        ↕              ↕            ↕                     │
│  ┌────────────────────────────────────────┐              │
│  │  SQLite Local Storage + JS Client UI  │              │
│  └────────────────────────────────────────┘              │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Federated Coordination                         │
│  ┌──────────────────┐  ┌─────────────────┐              │
│  │  Flower Client   │  │  Opacus DP-SGD  │              │
│  │  (flwr)          │  │  (noise+clip)   │              │
│  └──────────────────┘  └─────────────────┘              │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Central Server (Aggregation)                   │
│  ┌──────────────────┐  ┌─────────────────┐              │
│  │  FedAvg Server   │  │  Benchmarking   │              │
│  │  (Flower)        │  │  Engine         │              │
│  └──────────────────┘  └─────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
├── data/                    # Dataset download & preprocessing
├── models/                  # Layer 1 — On-device ML models
├── federated/               # Layer 2 — Flower + Opacus coordination
├── server/                  # Layer 3 — FL server, simulation, API
├── benchmarks/              # Evaluation & privacy-utility analysis
├── deployment/              # Model export (ExecuTorch / LiteRT)
├── client-app/              # JavaScript client-facing interface
└── tests/                   # Test suite
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & preprocess datasets
```bash
python data/download_datasets.py
python data/preprocess_tabular.py
python data/preprocess_skin.py
python data/preprocess_respiratory.py
```

### 3. Run federated simulation
```bash
# Tabular symptom model (MVP)
python server/run_simulation.py --model symptom_mlp --num_clients 10 --rounds 20 --epsilon 2.0

# Skin lesion model
python server/run_simulation.py --model skin_cnn --num_clients 5 --rounds 30 --epsilon 5.0

# Respiratory model
python server/run_simulation.py --model respiratory_cnn --num_clients 5 --rounds 30 --epsilon 5.0
```

### 4. Run baselines for comparison
```bash
python server/centralized_baseline.py --model symptom_mlp
python server/fl_no_dp_baseline.py --model symptom_mlp --num_clients 10 --rounds 20
```

### 5. Run benchmarks
```bash
python benchmarks/privacy_utility_sweep.py --model symptom_mlp
python benchmarks/plot_results.py
```

### 6. Launch the client app
```bash
# Start the API server
python server/api_server.py

# Open client-app/index.html in your browser
```

## 📊 Symptom Modalities

| Modality | Dataset | Model | Input |
|---|---|---|---|
| Tabular Symptoms | Kaggle Disease-Symptom [8] | MLP (3-layer) | Binary symptom vector |
| Skin Conditions | HAM10000 [6] | MobileNetV3-Small | 224×224 RGB image |
| Respiratory Sounds | ICBHI 2017 [7] | Small CNN | 128×128 Mel spectrogram |

## 🔒 Privacy Guarantees

- **Federated Learning**: Raw medical data **never leaves** the user's device
- **Differential Privacy**: Calibrated Gaussian noise via Opacus bounds information leakage
- **Privacy Budget (ε)**: Tunable and tracked across federated rounds
- **Gradient Clipping**: Per-sample gradient clipping prevents outlier exposure

## 👥 Team

| Member | Responsibility |
|---|---|
| Rohan Tiwari | Federated coordination & server aggregation (Flower / FedAvg) |
| Saar Ravindra Singh | Client-side model development & on-device deployment |
| Saksham Garg | Differential Privacy integration (Opacus) & benchmarking |

**Mentor**: Dr. Jaishree Jain, Department of CSE, AKGEC Ghaziabad

## 📚 References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017
2. Rieke et al., "The Future of Digital Health with Federated Learning," NPJ Digital Medicine 2020
3. Yousefpour et al., "Opacus: User-Friendly Differential Privacy Library in PyTorch," 2021
4. Dwork, "Differential Privacy," 2006
5. Beutel et al., "Flower: A Friendly Federated Learning Framework," 2020
6. Tschandl et al., "The HAM10000 Dataset," Scientific Data 2018
7. Rocha et al., "ICBHI 2017 Challenge Respiratory Sound Database," 2017
8. Kaggle Disease Symptom Prediction Dataset

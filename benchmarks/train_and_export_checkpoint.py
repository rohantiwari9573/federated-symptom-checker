"""
Trains the deployable SymptomMLP checkpoint served by server/api_server.py.

Reuses the same real dataset, real FedAvg (no-DP) training loop, and same
seed/round/client setup validated in real_comparison_experiment.py (the
script that produced benchmarks/results/comparison_results.csv and the
paper's Table II). FedAvg-no-DP is deployed (not a DP variant) because at
this project's benchmarked round budget, DP variants collapse accuracy
(see comparison_results.csv) while FedAvg-no-DP matches the centralized
ceiling -- so it is the honest, best-performing choice for the live demo.

Outputs (consumed by server/api_server.py):
  results/checkpoints/symptom_mlp_latest.npy    - state_dict tensors
  results/checkpoints/symptom_mlp_vocab.json    - ordered symptom feature names
  results/checkpoints/symptom_mlp_classes.json  - ordered disease class names
  results/checkpoints/symptom_mlp_metrics.json  - measured test accuracy/F1/loss
"""
import sys, json, random
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import pandas as pd
from models.symptom_mlp import SymptomMLP

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_CSV = PROJECT_ROOT / "data" / "raw" / "disease_symptom" / "dataset.csv"
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")

df = pd.read_csv(DATA_CSV)
df = df.dropna(axis=1, how="all")
label_col = df.columns[-1]
feature_cols = [c.strip() for c in df.columns if c != label_col]

X = df[[c for c in df.columns if c != label_col]].values.astype(np.float32)
labels_raw = df[label_col].astype(str).str.strip().values
classes = sorted(set(labels_raw))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[l] for l in labels_raw], dtype=np.int64)

num_features = X.shape[1]
num_classes = len(classes)
print(f"Loaded {len(X)} samples, {num_features} symptom features, {num_classes} disease classes")

idx = np.arange(len(X))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

NUM_CLIENTS = 10
ALPHA = 0.5
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.08

def dirichlet_partition(y, num_clients, alpha, seed=SEED):
    rng = np.random.default_rng(seed)
    nc = y.max() + 1
    client_indices = [[] for _ in range(num_clients)]
    for c in range(nc):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for cid, s in enumerate(splits):
            client_indices[cid].extend(s.tolist())
    return client_indices

client_indices = dirichlet_partition(y_train, NUM_CLIENTS, ALPHA)

def new_model():
    torch.manual_seed(SEED)
    return SymptomMLP(input_dim=num_features, num_classes=num_classes).to(device)

def get_params(model):
    return [p.detach().clone() for p in model.state_dict().values()]

def set_params(model, params):
    sd = model.state_dict()
    for k, p in zip(sd.keys(), params):
        sd[k] = p
    model.load_state_dict(sd)

def evaluate(model, X_t, y_t):
    model.eval()
    with torch.no_grad():
        out = model(X_t)
        preds = out.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
        loss = F.cross_entropy(out, y_t).item()
        f1s = []
        for c in range(num_classes):
            tp = ((preds == c) & (y_t == c)).sum().item()
            fp = ((preds == c) & (y_t != c)).sum().item()
            fn = ((preds != c) & (y_t == c)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))
    return acc, macro_f1, loss

def local_train(model, X_t, y_t, epochs, batch_size, lr):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = len(X_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_t[batch_idx], y_t[batch_idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
    return model

def fedavg(param_sets, weights):
    total = sum(weights)
    avg = []
    for tensors in zip(*param_sets):
        stacked = sum(t * (w / total) for t, w in zip(tensors, weights))
        avg.append(stacked)
    return avg

print("\n=== Training deployable FedAvg (no DP) global model ===")
global_model = new_model()
for r in range(ROUNDS):
    client_params, client_weights = [], []
    for cid, idxs in enumerate(client_indices):
        if len(idxs) < 2:
            continue
        local_model = new_model()
        set_params(local_model, get_params(global_model))
        xb, yb = X_train_t[idxs], y_train_t[idxs]
        local_train(local_model, xb, yb, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        client_params.append(get_params(local_model))
        client_weights.append(len(idxs))
    set_params(global_model, fedavg(client_params, client_weights))
    acc, f1, loss = evaluate(global_model, X_test_t, y_test_t)
    print(f"round {r+1:2d}  acc={acc:.4f}  f1={f1:.4f}  loss={loss:.4f}")

final_acc, final_f1, final_loss = evaluate(global_model, X_test_t, y_test_t)
print(f"\nFinal held-out test accuracy: {final_acc:.4f}  macro-F1: {final_f1:.4f}  loss: {final_loss:.4f}")

# --- Export checkpoint + vocab + classes + metrics ---
state = global_model.state_dict()
params = [v.cpu().numpy() for v in state.values()]
np.save(CKPT_DIR / "symptom_mlp_latest.npy", np.array(params, dtype=object), allow_pickle=True)

with open(CKPT_DIR / "symptom_mlp_vocab.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

with open(CKPT_DIR / "symptom_mlp_classes.json", "w") as f:
    json.dump(classes, f, indent=2)

with open(CKPT_DIR / "symptom_mlp_metrics.json", "w") as f:
    json.dump({
        "method": "Federated Averaging (FedAvg), no differential privacy",
        "rounds": ROUNDS,
        "num_clients": NUM_CLIENTS,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(final_acc, 4),
        "macro_f1": round(final_f1, 4),
        "loss": round(final_loss, 4),
        "note": "Measured on a held-out test split of a public Kaggle disease-symptom "
                "dataset (near-deterministic symptom-to-disease mapping); this is "
                "benchmark accuracy on that dataset, not clinical diagnostic accuracy."
    }, f, indent=2)

print(f"\nSaved checkpoint + vocab + classes + metrics -> {CKPT_DIR}")

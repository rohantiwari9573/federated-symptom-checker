"""
Real experimental comparison for the conference paper's Results section.

Trains the project's actual SymptomMLP architecture (models/symptom_mlp.py)
on the real Kaggle Disease-Symptom-Description dataset [8] under three
regimes and logs genuine measured metrics (no mocked numbers):

  1. Centralized  - single model trained on the full pooled training set.
  2. Federated (FedAvg, no DP) - McMahan et al. [1] averaging across
     10 simulated non-IID clients.
  3. Federated + DP (FedAvg + manual per-sample gradient clipping and
     calibrated Gaussian noise, i.e. the DP-SGD mechanism Opacus [3]
     implements) at several privacy budgets (epsilon).

Outputs:
  benchmarks/results/comparison_results.csv
  benchmarks/results/accuracy_vs_rounds.png
  benchmarks/results/privacy_utility_tradeoff.png
  benchmarks/results/final_accuracy_bar.png
"""
import sys, os, csv, math, random, copy
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.symptom_mlp import SymptomMLP
from federated.dp_config import create_dp_config

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_CSV = PROJECT_ROOT / "data" / "raw" / "disease_symptom" / "dataset.csv"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Load & preprocess the real dataset
# ---------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv(DATA_CSV)
df = df.dropna(axis=1, how="all")  # drop the trailing empty column from the CSV
label_col = df.columns[-1]
feature_cols = [c for c in df.columns if c != label_col]

X = df[feature_cols].values.astype(np.float32)
labels_raw = df[label_col].astype(str).str.strip().values
classes = sorted(set(labels_raw))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[l] for l in labels_raw], dtype=np.int64)

num_features = X.shape[1]
num_classes = len(classes)
print(f"Loaded {len(X)} samples, {num_features} symptom features, {num_classes} disease classes")

# Shuffle + split (80/20 stratified-ish by simple shuffle since dataset is balanced per class)
idx = np.arange(len(X))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

# ---------------------------------------------------------------------------
# 2. Non-IID client partitioning (Dirichlet split, as flwr-datasets would do)
# ---------------------------------------------------------------------------
NUM_CLIENTS = 10
ALPHA = 0.5  # Dirichlet concentration -> lower = more non-IID

def dirichlet_partition(y, num_clients, alpha, seed=SEED):
    rng = np.random.default_rng(seed)
    num_classes = y.max() + 1
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        for cid, s in enumerate(splits):
            client_indices[cid].extend(s.tolist())
    return client_indices

client_indices = dirichlet_partition(y_train, NUM_CLIENTS, ALPHA)
print("Client shard sizes:", [len(c) for c in client_indices])

# ---------------------------------------------------------------------------
# 3. Shared helpers
# ---------------------------------------------------------------------------
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
        # macro F1
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

def local_train(model, X_t, y_t, epochs, batch_size, lr, dp=None):
    """
    dp: None or dict(max_grad_norm=float, noise_multiplier=float)
    When dp is set, applies per-sample gradient clipping + Gaussian noise
    every step -- the same DP-SGD mechanism Opacus [3] implements internally.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    n = len(X_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_t[batch_idx], y_t[batch_idx]
            if not dp:
                opt.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.step()
            else:
                # Per-sample gradients via microbatching (manual DP-SGD)
                opt.zero_grad()
                param_list = [p for p in model.parameters() if p.requires_grad]
                summed_grads = [torch.zeros_like(p) for p in param_list]
                for i in range(len(xb)):
                    model.zero_grad()
                    out = model(xb[i:i+1])
                    loss = F.cross_entropy(out, yb[i:i+1])
                    grads = torch.autograd.grad(loss, param_list)
                    total_norm = math.sqrt(sum(g.norm(2).item() ** 2 for g in grads))
                    clip_factor = min(1.0, dp['max_grad_norm'] / (total_norm + 1e-6))
                    for sg, g in zip(summed_grads, grads):
                        sg.add_(g * clip_factor)
                noise_std = dp['noise_multiplier'] * dp['max_grad_norm']
                with torch.no_grad():
                    for p, sg in zip(param_list, summed_grads):
                        noise = torch.normal(0.0, noise_std, size=sg.shape)
                        p.grad = (sg + noise) / len(xb)
                opt.step()
    return model

def fedavg(param_sets, weights):
    total = sum(weights)
    avg = []
    for tensors in zip(*param_sets):
        stacked = sum(t * (w / total) for t, w in zip(tensors, weights))
        avg.append(stacked)
    return avg

# Real noise multiplier via the project's own DP configuration helper
# (federated/dp_config.py), which uses Opacus's RDP accountant -- the
# same mechanism McMahan's FedAvg [1] is composed with per client.
_noise_cache = {}
def client_noise_multiplier(epsilon, delta, num_samples, local_epochs, batch_size):
    key = (epsilon, num_samples)
    if key not in _noise_cache:
        cfg = create_dp_config(
            epsilon=epsilon, delta=delta, max_grad_norm=1.0,
            num_train_samples=num_samples, epochs=local_epochs * ROUNDS,
            batch_size=batch_size,
        )
        _noise_cache[key] = cfg.noise_multiplier
    return _noise_cache[key]

# ---------------------------------------------------------------------------
# 4. Centralized baseline
# ---------------------------------------------------------------------------
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.08

print("\n=== Centralized baseline ===")
cent_model = new_model()
cent_curve = []
for r in range(ROUNDS):
    local_train(cent_model, X_train_t, y_train_t, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    acc, f1, loss = evaluate(cent_model, X_test_t, y_test_t)
    cent_curve.append(acc)
    print(f"round {r+1:2d}  acc={acc:.4f}  f1={f1:.4f}  loss={loss:.4f}")
cent_final_acc, cent_final_f1, cent_final_loss = evaluate(cent_model, X_test_t, y_test_t)

# ---------------------------------------------------------------------------
# 5. Federated, no DP (FedAvg)
# ---------------------------------------------------------------------------
def run_federated(epsilon=None, tag="fed"):
    print(f"\n=== Federated run: {tag} ===")
    global_model = new_model()
    curve = []
    for r in range(ROUNDS):
        client_params, client_weights = [], []
        for cid, idxs in enumerate(client_indices):
            if len(idxs) < 2:
                continue
            local_model = new_model()
            set_params(local_model, get_params(global_model))
            xb = X_train_t[idxs]
            yb = y_train_t[idxs]
            dp_settings = None
            if epsilon is not None:
                nm = client_noise_multiplier(epsilon, DELTA, len(idxs), LOCAL_EPOCHS, BATCH_SIZE)
                dp_settings = {'max_grad_norm': 1.0, 'noise_multiplier': nm}
            local_train(local_model, xb, yb, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, lr=LR, dp=dp_settings)
            client_params.append(get_params(local_model))
            client_weights.append(len(idxs))
        new_global_params = fedavg(client_params, client_weights)
        set_params(global_model, new_global_params)
        acc, f1, loss = evaluate(global_model, X_test_t, y_test_t)
        curve.append(acc)
        print(f"round {r+1:2d}  acc={acc:.4f}  f1={f1:.4f}  loss={loss:.4f}")
    facc, ff1, floss = evaluate(global_model, X_test_t, y_test_t)
    final_global_models[tag] = global_model
    return curve, facc, ff1, floss

DELTA = 1e-5
final_global_models = {}
fed_curve, fed_acc, fed_f1, fed_loss = run_federated(epsilon=None, tag="FedAvg (no DP)")

# ---------------------------------------------------------------------------
# 6. Federated + DP at several epsilon budgets
# ---------------------------------------------------------------------------
epsilons = [0.5, 1.0, 2.0, 5.0]
dp_results = {}
dp_curves = {}
for eps in epsilons:
    curve, acc, f1, loss = run_federated(epsilon=eps, tag=f"FedAvg+DP eps={eps}")
    dp_results[eps] = (acc, f1, loss)
    dp_curves[eps] = curve

# ---------------------------------------------------------------------------
# 7. Save CSV
# ---------------------------------------------------------------------------
csv_path = OUT_DIR / "comparison_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "epsilon", "final_accuracy", "final_macro_f1", "final_loss"])
    writer.writerow(["centralized", "n/a", f"{cent_final_acc:.4f}", f"{cent_final_f1:.4f}", f"{cent_final_loss:.4f}"])
    writer.writerow(["federated_no_dp", "inf", f"{fed_acc:.4f}", f"{fed_f1:.4f}", f"{fed_loss:.4f}"])
    for eps in epsilons:
        acc, f1, loss = dp_results[eps]
        writer.writerow(["federated_dp", eps, f"{acc:.4f}", f"{f1:.4f}", f"{loss:.4f}"])
print(f"\nSaved results table -> {csv_path}")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None

# --- Accuracy vs communication round ---
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
rounds_x = list(range(1, ROUNDS + 1))
ax.plot(rounds_x, [cent_final_acc]*ROUNDS if False else cent_curve, label="Centralized (upper bound)", linewidth=2, color="#1f6f54")
ax.plot(rounds_x, fed_curve, label="Federated, FedAvg (no DP)", linewidth=2, color="#2563EB")
colors = ["#7C3AED", "#D97706", "#DC2626", "#059669"]
for c, eps in zip(colors, epsilons):
    ax.plot(rounds_x, dp_curves[eps], label=f"Federated + DP (ε={eps})", linewidth=1.6, linestyle="--", color=c)
ax.set_xlabel("Federated communication round")
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy vs. communication round")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "accuracy_vs_rounds.png")
plt.close(fig)

# --- Privacy-utility tradeoff ---
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)
xs = epsilons + [10.0]  # extend visually toward "no privacy"
ys = [dp_results[e][0] for e in epsilons] + [fed_acc]
ax.plot(xs, ys, marker="o", color="#DC2626", linewidth=2)
ax.axhline(cent_final_acc, color="#1f6f54", linestyle=":", linewidth=1.5, label="Centralized baseline")
ax.axhline(fed_acc, color="#2563EB", linestyle=":", linewidth=1.5, label="Federated, no DP")
ax.set_xlabel("Privacy budget ε (lower = stronger privacy)")
ax.set_ylabel("Test accuracy")
ax.set_title("Privacy-utility trade-off")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "privacy_utility_tradeoff.png")
plt.close(fig)

# --- Final accuracy bar comparison ---
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
labels = ["Centralized"] + ["FedAvg\n(no DP)"] + [f"FedAvg+DP\nε={e}" for e in epsilons]
values = [cent_final_acc, fed_acc] + [dp_results[e][0] for e in epsilons]
bar_colors = ["#1f6f54", "#2563EB"] + ["#DC2626"]*len(epsilons)
bars = ax.bar(labels, values, color=bar_colors)
ax.set_ylabel("Final test accuracy")
ax.set_title("Final accuracy: centralized vs. federated vs. federated+DP")
ax.set_ylim(0, 1.0)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
plt.xticks(fontsize=7)
fig.tight_layout()
fig.savefig(OUT_DIR / "final_accuracy_bar.png")
plt.close(fig)

print("Saved plots to", OUT_DIR)

# ---------------------------------------------------------------------------
# 8. Export the eps=5.0 DP global model for in-browser inference
#    (client-app/model/). Same weights as the Table II row for eps=5.0.
# ---------------------------------------------------------------------------
import json
MODEL_EXPORT_DIR = PROJECT_ROOT / "client-app" / "model"
MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
export_model = final_global_models["FedAvg+DP eps=5.0"]
export_weights = {
    name: tensor.detach().cpu().numpy().round(6).tolist()
    for name, tensor in export_model.state_dict().items()
}
export_payload = {
    "source": "benchmarks/real_comparison_experiment.py",
    "config": {"epsilon": 5.0, "delta": DELTA, "rounds": ROUNDS,
               "local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE,
               "num_clients": NUM_CLIENTS, "alpha": ALPHA, "seed": SEED},
    "test_accuracy": dp_results[5.0][0],
    "test_macro_f1": dp_results[5.0][1],
    "vocab": list(feature_cols),
    "classes": classes,
    "weights": export_weights,
}
with open(MODEL_EXPORT_DIR / "symptom_mlp_eps5.json", "w") as f:
    json.dump(export_payload, f)
print("Exported browser model ->", MODEL_EXPORT_DIR / "symptom_mlp_eps5.json")

# Test set for parity checking the JS forward pass against PyTorch.
with open(MODEL_EXPORT_DIR / "parity_test_set.json", "w") as f:
    json.dump({"X": X_test.astype(int).tolist(),
               "y_true": y_test.tolist(),
               "y_torch_logits": export_model.eval()(X_test_t).detach().numpy().round(6).tolist()}, f)
print("\nDONE.")

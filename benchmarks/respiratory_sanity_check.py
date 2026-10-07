"""
Centralized sanity check for the respiratory model (no federation, no DP).

Same purpose as skin_sanity_check.py: find out whether RespiratoryCNN can learn
ICBHI at all, and which settings matter, before any federated result is read.

Protocol:
  - Patient-level 80/20 split (same as the federated experiment, seed 42).
  - Configs are compared on a validation split carved from training patients.
  - The test split is used ONCE, with --final_test, for the chosen config.

Run:  python benchmarks/respiratory_sanity_check.py --config adam_weighted --epochs 15 --final_test
"""
import sys, json, time, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from models.respiratory_cnn import RespiratoryCNN
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
from respiratory_federated_experiment import (
    build_dataset, patient_split, evaluate, CLASSES, SEED)

CACHE = PROJECT_ROOT / "data" / "processed" / "icbhi_segments.pt"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "respiratory"

CONFIGS = {
    # name: (optimizer, lr, class_weighted)
    "sgd_plain":     ("sgd",  0.01, False),
    "adam_plain":    ("adam", 1e-3, False),
    "adam_weighted": ("adam", 1e-3, True),
    "adam_lowlr":    ("adam", 3e-4, True),
}


def load_segments():
    """Build the 3 s log-mel segments once and cache them (the decode step is slow)."""
    if CACHE.exists():
        c = torch.load(CACHE, weights_only=False)
        return c["X"], c["y"], c["pid"]
    X, y, pid = build_dataset()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "y": y, "pid": pid}, CACHE)
    return X, y, pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--final_test", action="store_true")
    args = ap.parse_args()
    opt_name, lr, weighted = CONFIGS[args.config]

    torch.manual_seed(SEED); np.random.seed(SEED)
    X, y, pid = load_segments()
    idx_tr, idx_te = patient_split(pid)

    # Validation split by patient, carved from the training patients only.
    rng = np.random.default_rng(SEED)
    tr_patients = np.unique(pid[idx_tr])
    rng.shuffle(tr_patients)
    val_p = set(tr_patients[: max(1, len(tr_patients) // 8)])
    is_val = np.array([p in val_p for p in pid[idx_tr]])
    idx_fit, idx_val = idx_tr[~is_val], idx_tr[is_val]

    Xf, yf = X[idx_fit], y[idx_fit]
    Xv, yv = X[idx_val], y[idx_val]
    print(f"config={args.config} opt={opt_name} lr={lr} weighted={weighted}", flush=True)
    print(f"fit={len(yf)} val={len(yv)} test={len(idx_te)}", flush=True)
    print("fit class counts:", torch.bincount(yf, minlength=4).tolist(), flush=True)

    torch.manual_seed(SEED)
    model = RespiratoryCNN(num_classes=len(CLASSES))
    opt = (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if opt_name == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr))

    weights = None
    if weighted:
        counts = torch.bincount(yf, minlength=4).float()
        weights = (counts.sum() / (4 * counts)).clamp(max=10.0)
        print("class weights:", [round(w, 2) for w in weights.tolist()], flush=True)

    history = []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(yf))
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            loss = F.cross_entropy(model(Xf[b]), yf[b], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        va_acc, va_f1, _ = evaluate(model, Xv, yv)
        history.append({"epoch": ep + 1, "val_acc": va_acc, "val_macro_f1": va_f1})
        print(f"epoch {ep+1:2d}  loss={loss.item():.3f}  val_acc={va_acc:.4f}  "
              f"val_macro_f1={va_f1:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    result = {"config": args.config, "history": history}
    if args.final_test:
        te_acc, te_f1, _ = evaluate(model, X[idx_te], y[idx_te])
        result["test_acc"], result["test_macro_f1"] = te_acc, te_f1
        print(f"TEST  acc={te_acc:.4f}  macro_f1={te_f1:.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sanity_{args.config}{'_final' if args.final_test else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

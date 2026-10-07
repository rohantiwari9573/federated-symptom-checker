"""
Centralized sanity check for the skin model (no federation, no DP).

Purpose: find out whether SkinCNN (pretrained MobileNetV3-Small) can learn
HAM10000 at all, and which training settings matter.

Protocol:
  - Same 80/20 stratified split as the federated experiment (seed 42).
  - The 20% test split is used ONCE, for the final report of the chosen config.
  - Configs are compared on a 10% validation split carved from the training set.
  - Keeps the pretrained BatchNorm (no GroupNorm conversion, which is only
    needed for Opacus DP-SGD and discards pretrained statistics).

Run:  python benchmarks/skin_sanity_check.py --config adam_weighted --epochs 6
"""
import sys, time, json, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pandas as pd

from models.skin_cnn import SkinCNN

SEED = 42
CACHE = PROJECT_ROOT / "data" / "processed" / "ham10000_224.pt"
META = PROJECT_ROOT / "data" / "raw" / "ham10000_meta" / "HAM10000_metadata.csv"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "skin"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CONFIGS = {
    # name: (optimizer, lr, class_weighted, augment)
    "sgd_plain":      ("sgd",  0.01, False, False),
    "adam_plain":     ("adam", 1e-3, False, False),
    "adam_weighted":  ("adam", 1e-3, True,  True),
    "adam_lowlr":     ("adam", 3e-4, True,  True),
    "adam_cosine":    ("adam", 3e-4, True,  True),  # same as adam_lowlr, with cosine LR decay
    "adam_cosine60":  ("adam", 3e-4, True,  True),  # cosine, longer schedule
}


def normalize(x):
    return (x.float() / 255.0 - MEAN) / STD


def augment(x):
    """Dermoscopy images have no canonical orientation: flips and 90-degree rotations are label-preserving."""
    if np.random.rand() < 0.5:
        x = torch.flip(x, dims=[3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, dims=[2])
    k = np.random.randint(4)
    return torch.rot90(x, k, dims=[2, 3])


@torch.inference_mode()
def evaluate(model, X, y, bs=128):
    model.eval()
    preds = []
    for s in range(0, len(X), bs):
        preds.append(model(normalize(X[s:s + bs])).argmax(1))
    p = torch.cat(preds).numpy()
    t = y.numpy()
    return float((p == t).mean()), float(f1_score(t, p, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grouped", action=argparse.BooleanOptionalAction, default=True,
                    help="split by lesion_id (default); --no-grouped reproduces the old random split")
    ap.add_argument("--final_test", action="store_true",
                    help="evaluate on the held-out test split (do this once, for the chosen config)")
    args = ap.parse_args()
    opt_name, lr, weighted, aug = CONFIGS[args.config]

    torch.manual_seed(SEED); np.random.seed(SEED)
    cache = torch.load(CACHE, weights_only=False)  # our own cache file (contains numpy labels)
    X, labels = cache["X"], np.asarray(cache["labels"])
    if args.grouped:
        # Split by lesion: several images of one lesion must never sit on both sides.
        meta = pd.read_csv(META)
        lesion = dict(zip(meta["image_id"], meta["lesion_id"]))
        groups = np.array([lesion[i] for i in cache["ids"]])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        idx_tr, idx_te = next(gss.split(np.zeros(len(labels)), labels, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        a, b = next(gss2.split(np.zeros(len(idx_tr)), labels[idx_tr], groups[idx_tr]))
        idx_fit, idx_val = idx_tr[a], idx_tr[b]
    else:
        idx_tr, idx_te = train_test_split(np.arange(len(labels)), test_size=0.2,
                                          stratify=labels, random_state=SEED)
        idx_fit, idx_val = train_test_split(idx_tr, test_size=0.1,
                                            stratify=labels[idx_tr], random_state=SEED)
    y = torch.tensor(labels, dtype=torch.long)

    Xf, yf = X[idx_fit], y[idx_fit]
    Xv, yv = X[idx_val], y[idx_val]
    print(f"config={args.config} opt={opt_name} lr={lr} weighted={weighted} aug={aug}", flush=True)
    print(f"fit={len(yf)} val={len(yv)} test={len(idx_te)}", flush=True)
    print("fit class counts:", torch.bincount(yf, minlength=7).tolist(), flush=True)

    model = SkinCNN(num_classes=7)
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    weights = None
    if weighted:
        counts = torch.bincount(yf, minlength=7).float()
        weights = (counts.sum() / (7 * counts)).clamp(max=10.0)
        print("class weights:", [round(w, 2) for w in weights.tolist()], flush=True)

    sched = None
    if args.config.startswith("adam_cosine"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(yf))
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            xb = Xf[b]
            if aug:
                xb = augment(xb)
            loss = F.cross_entropy(model(normalize(xb)), yf[b], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        va_acc, va_f1 = evaluate(model, Xv, yv)
        history.append({"epoch": ep + 1, "val_acc": va_acc, "val_macro_f1": va_f1})
        print(f"epoch {ep+1}  loss={loss.item():.3f}  val_acc={va_acc:.4f}  val_macro_f1={va_f1:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    result = {"config": args.config, "history": history}
    if args.final_test:
        te_acc, te_f1 = evaluate(model, X[idx_te], y[idx_te])
        result["test_acc"], result["test_macro_f1"] = te_acc, te_f1
        print(f"TEST  acc={te_acc:.4f}  macro_f1={te_f1:.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "grouped" if args.grouped else "random"
    out = OUT_DIR / f"sanity_{args.config}_{tag}{'_final' if args.final_test else ''}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

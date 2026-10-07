"""
Federated skin-lesion experiment on HAM10000 (paper Section V-D future work).

Three regimes, mirroring the tabular experiment:
  - FedAvg without DP
  - FedAvg + DP-SGD (Opacus PrivacyEngine) at eps in {0.5, 1.0, 2.0, 5.0}

Model: SkinCNN (MobileNetV3-Small, ImageNet-pretrained), with BatchNorm
replaced by GroupNorm via Opacus ModuleValidator.fix so per-sample gradients
can be computed. Data: data/raw/ham10000 (GroundTruth.csv + images/).

Run:
  python benchmarks/skin_federated_experiment.py            # full run (hours on CPU)
  python benchmarks/skin_federated_experiment.py --smoke    # tiny sanity run
"""
import sys, os, csv, copy, json, time, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from torchvision.models.mobilenetv3 import InvertedResidual
from models.skin_cnn import SkinCNN
from federated.dp_config import create_dp_config

SEED = 42
RAW = PROJECT_ROOT / "data" / "raw" / "ham10000"
CACHE = PROJECT_ROOT / "data" / "processed" / "ham10000_224.pt"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "skin"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_images(ids, size=224):
    """Decode and resize to uint8 [N,3,H,W]. Fails loudly if an image is missing."""
    arr = np.empty((len(ids), 3, size, size), dtype=np.uint8)
    for i, img_id in enumerate(ids):
        path = RAW / "images" / f"{img_id}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"missing image {path}")
        im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        arr[i] = np.asarray(im).transpose(2, 0, 1)
    return torch.from_numpy(arr)


def prepare(smoke):
    gt = pd.read_csv(RAW / "GroundTruth.csv")
    assert list(gt.columns[1:]) == CLASSES, f"unexpected columns {list(gt.columns)}"
    labels = gt[CLASSES].values.argmax(axis=1)
    ids = gt["image"].tolist()
    if smoke:
        ids, labels = ids[:400], labels[:400]
    idx_tr, idx_te = train_test_split(
        np.arange(len(ids)), test_size=0.2, stratify=labels, random_state=SEED)
    if CACHE.exists() and not smoke:
        cache = torch.load(CACHE, weights_only=False)
        X = cache["X"]
    else:
        print(f"Decoding {len(ids)} images (one-time, ~minutes)...", flush=True)
        X = load_images(ids)
        if not smoke:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"X": X, "ids": ids, "labels": labels}, CACHE)
    y = torch.tensor(labels, dtype=torch.long)
    return X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]


def normalize(x_uint8):
    return (x_uint8.float() / 255.0 - MEAN) / STD


class _SafeInvertedResidual(InvertedResidual):
    """torchvision's InvertedResidual does `result += input`, which Opacus rejects."""
    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


def augment(x):
    """Flips and 90-degree rotations: label-preserving for dermoscopy images."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    return torch.rot90(x, int(torch.randint(4, (1,)).item()), dims=[2, 3])


def new_model(dp):
    """BatchNorm is kept for the non-DP baseline (pretrained statistics survive).
    Only DP runs convert to GroupNorm, because Opacus cannot handle BatchNorm."""
    torch.manual_seed(SEED)
    m = SkinCNN(num_classes=len(CLASSES))
    # Opacus cannot handle in-place activations, in-place residual adds, or BatchNorm.
    for mod in m.modules():
        if isinstance(mod, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid)):
            mod.inplace = False
        if type(mod) is InvertedResidual:
            mod.__class__ = _SafeInvertedResidual
    return ModuleValidator.fix(m) if dp else m


def dirichlet_partition(y, num_clients, alpha, seed=SEED):
    rng = np.random.default_rng(seed)
    yn = y.numpy()
    client_idx = [[] for _ in range(num_clients)]
    for c in np.unique(yn):
        idx_c = np.where(yn == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx_c, cuts)):
            client_idx[cid].extend(part.tolist())
    return [np.array(ix, dtype=np.int64) for ix in client_idx]


@torch.inference_mode()
def evaluate(model, X, y, bs=128):
    model.eval()
    preds, losses = [], []
    for s in range(0, len(X), bs):
        out = model(normalize(X[s:s + bs]))
        losses.append(F.cross_entropy(out, y[s:s + bs], reduction="sum").item())
        preds.append(out.argmax(1))
    p = torch.cat(preds).numpy()
    t = y.numpy()
    return float((p == t).mean()), float(f1_score(t, p, average="macro")), sum(losses) / len(t)


def local_train(global_model, Xc, yc, args, dp_cfg, gen, class_w):
    """One client's local round. With dp_cfg enabled, runs Opacus DP-SGD."""
    model = copy.deepcopy(global_model)
    model.train()
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xc, yc), batch_size=bs, shuffle=True, generator=gen)
    if dp_cfg is not None and dp_cfg.enabled:
        pe = PrivacyEngine()
        model, opt, loader = pe.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=dp_cfg.noise_multiplier, max_grad_norm=dp_cfg.max_grad_norm)
    for _ in range(args.local_epochs):
        for xb, yb in loader:
            if args.augment:
                xb = augment(xb)
            opt.zero_grad()
            loss = F.cross_entropy(model(normalize(xb)), yb, weight=class_w)
            loss.backward()
            opt.step()
    module = model._module if hasattr(model, "_module") else model
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def fedavg(states, weights):
    total = float(sum(weights))
    return {k: sum(s[k] * (w / total) for s, w in zip(states, weights)) for k in states[0]}


def run(tag, epsilon, args, Xtr, ytr, Xte, yte, parts, log):
    print(f"\n=== {tag} ===", flush=True)
    gen = torch.Generator().manual_seed(SEED)
    global_model = new_model(dp=epsilon is not None)
    class_w = None
    if args.class_weighted:
        counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
        class_w = (counts.sum() / (len(CLASSES) * counts)).clamp(max=10.0)
    curve = []
    for r in range(args.rounds):
        t0 = time.time()
        states, weights = [], []
        for cid, idx in enumerate(parts):
            if len(idx) < 2:
                continue
            dp_cfg = None
            if epsilon is not None:
                # Per-client calibration (paper Section III-C); composition over all rounds (V-D).
                dp_cfg = create_dp_config(
                    epsilon=epsilon, delta=args.delta, max_grad_norm=1.0,
                    num_train_samples=len(idx),
                    epochs=args.local_epochs * args.rounds, batch_size=args.batch_size)
            st = local_train(global_model, Xtr[idx], ytr[idx], args, dp_cfg, gen, class_w)
            states.append(st)
            weights.append(len(idx))
        global_model.load_state_dict(fedavg(states, weights), strict=False)
        acc, f1, loss = evaluate(global_model, Xte, yte)
        curve.append(acc)
        msg = f"round {r+1:2d}  acc={acc:.4f}  f1={f1:.4f}  loss={loss:.4f}  ({time.time()-t0:.0f}s)"
        print(msg, flush=True)
        log.write(f"{tag},{r+1},{acc:.6f},{f1:.6f},{loss:.6f}\n"); log.flush()
    return acc, f1, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--class_weighted", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--epsilons", type=str, default="0.5,1.0,2.0,5.0")
    args = ap.parse_args()
    if args.smoke:
        args.rounds = 1
        args.clients = 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = prepare(args.smoke)
    print(f"train={len(ytr)} test={len(yte)} classes={len(CLASSES)}", flush=True)
    parts = dirichlet_partition(ytr, args.clients, args.alpha)
    print("client shard sizes:", [len(p) for p in parts], flush=True)

    out_path = OUT_DIR / ("smoke_curve.csv" if args.smoke else "skin_curves.csv")
    results = {}
    with open(out_path, "w") as log:
        log.write("method,round,accuracy,macro_f1,loss\n")
        results["FedAvg (no DP)"] = run("FedAvg (no DP)", None, args, Xtr, ytr, Xte, yte, parts, log)
        for eps in [float(e) for e in args.epsilons.split(",")]:
            tag = f"FedAvg+DP eps={eps}"
            results[tag] = run(tag, eps, args, Xtr, ytr, Xte, yte, parts, log)

    summary = OUT_DIR / ("smoke_summary.json" if args.smoke else "skin_summary.json")
    with open(summary, "w") as f:
        json.dump({"config": vars(args), "train": len(ytr), "test": len(yte),
                   "final": {k: {"accuracy": v[0], "macro_f1": v[1], "loss": v[2]}
                             for k, v in results.items()}}, f, indent=2)
    print("\nWrote", out_path, "and", summary, flush=True)


if __name__ == "__main__":
    main()

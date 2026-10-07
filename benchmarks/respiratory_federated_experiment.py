"""
Federated respiratory-sound experiment on ICBHI 2017 (paper Section V-D future work).

Same three regimes as the tabular and skin experiments:
  - FedAvg without DP
  - FedAvg + DP-SGD (Opacus PrivacyEngine) at eps in {0.5, 1.0, 2.0, 5.0}

Model: RespiratoryCNN (GroupNorm-based, Opacus-compatible).
Input: 3-second log-mel spectrograms, 128 x 128, cut from each recording.
Labels: Normal / Crackle / Wheeze / Both, from the per-cycle annotations.
Split: by patient, so no patient appears in both train and test.

Run:
  python benchmarks/respiratory_federated_experiment.py
"""
import sys, copy, json, time, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from opacus import PrivacyEngine

from models.respiratory_cnn import RespiratoryCNN
from federated.dp_config import create_dp_config

SEED = 42
RAW = PROJECT_ROOT / "data" / "raw" / "icbhi" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database" / "audio_and_txt_files"
OUT_DIR = PROJECT_ROOT / "benchmarks" / "results" / "respiratory"
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]
SR, SEG_SEC, N_MELS, N_FRAMES = 22050, 3.0, 128, 128


def recording_label(txt_path):
    """ICBHI per-cycle rows: start, end, crackle flag, wheeze flag."""
    cyc = pd.read_csv(txt_path, sep=r"\s+", header=None, names=["start", "end", "crackle", "wheeze"])
    c, w = cyc["crackle"].sum() > 0, cyc["wheeze"].sum() > 0
    if c and w:
        return 3
    if w:
        return 2
    if c:
        return 1
    return 0


def segments_for(wav_path):
    """Cut a recording into non-overlapping 3 s log-mel segments [n, 1, 128, 128]."""
    y, _ = librosa.load(wav_path, sr=SR)
    seg_len = int(SR * SEG_SEC)
    n_seg = max(1, int(np.ceil(len(y) / seg_len)))
    y = np.pad(y, (0, n_seg * seg_len - len(y)))
    out = []
    for s in range(n_seg):
        chunk = y[s * seg_len:(s + 1) * seg_len]
        mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, hop_length=seg_len // N_FRAMES)
        mel = librosa.power_to_db(mel, ref=np.max)[:, :N_FRAMES]
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        out.append(mel)
    return np.stack(out)[:, None, :, :].astype(np.float32)


def build_dataset():
    wavs = sorted(RAW.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no wav files in {RAW}")
    rows_x, rows_y, rows_pid = [], [], []
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(f"missing annotation {txt}")
        label = recording_label(txt)
        patient = int(wav.name.split("_")[0])
        segs = segments_for(wav)
        rows_x.append(segs)
        rows_y.extend([label] * len(segs))
        rows_pid.extend([patient] * len(segs))
    X = torch.from_numpy(np.concatenate(rows_x))
    y = torch.tensor(rows_y, dtype=torch.long)
    pid = np.array(rows_pid)
    return X, y, pid


def patient_split(pid, test_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    patients = np.unique(pid)
    rng.shuffle(patients)
    n_test = int(round(len(patients) * test_frac))
    test_p = set(patients[:n_test])
    te = np.array([p in test_p for p in pid])
    return np.where(~te)[0], np.where(te)[0]


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


def new_model():
    torch.manual_seed(SEED)
    return RespiratoryCNN(num_classes=len(CLASSES))


@torch.inference_mode()
def evaluate(model, X, y, bs=256):
    model.eval()
    preds, loss_sum = [], 0.0
    for s in range(0, len(X), bs):
        out = model(X[s:s + bs])
        loss_sum += F.cross_entropy(out, y[s:s + bs], reduction="sum").item()
        preds.append(out.argmax(1))
    p = torch.cat(preds).numpy()
    t = y.numpy()
    return float((p == t).mean()), float(f1_score(t, p, average="macro")), loss_sum / len(t)


def local_train(global_model, Xc, yc, epochs, bs, lr, dp_cfg, gen):
    model = copy.deepcopy(global_model)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xc, yc), batch_size=bs, shuffle=True, generator=gen)
    if dp_cfg is not None and dp_cfg.enabled:
        pe = PrivacyEngine()
        model, opt, loader = pe.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=dp_cfg.noise_multiplier, max_grad_norm=dp_cfg.max_grad_norm)
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    module = model._module if hasattr(model, "_module") else model
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def fedavg(states, weights):
    total = float(sum(weights))
    return {k: sum(s[k] * (w / total) for s, w in zip(states, weights)) for k in states[0]}


def run(tag, epsilon, args, Xtr, ytr, Xte, yte, parts, log):
    print(f"\n=== {tag} ===", flush=True)
    gen = torch.Generator().manual_seed(SEED)
    global_model = new_model()
    for r in range(args.rounds):
        t0 = time.time()
        states, weights = [], []
        for idx in parts:
            if len(idx) < 2:
                continue
            dp_cfg = None
            if epsilon is not None:
                dp_cfg = create_dp_config(
                    epsilon=epsilon, delta=args.delta, max_grad_norm=1.0,
                    num_train_samples=len(idx),
                    epochs=args.local_epochs * args.rounds, batch_size=args.batch_size)
            states.append(local_train(global_model, Xtr[idx], ytr[idx], args.local_epochs,
                                      args.batch_size, args.lr, dp_cfg, gen))
            weights.append(len(idx))
        global_model.load_state_dict(fedavg(states, weights), strict=False)
        acc, f1, loss = evaluate(global_model, Xte, yte)
        print(f"round {r+1:2d}  acc={acc:.4f}  f1={f1:.4f}  loss={loss:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        log.write(f"{tag},{r+1},{acc:.6f},{f1:.6f},{loss:.6f}\n"); log.flush()
    return evaluate(global_model, Xte, yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--epsilons", type=str, default="0.5,1.0,2.0,5.0")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building ICBHI segments (one-time)...", flush=True)
    X, y, pid = build_dataset()
    idx_tr, idx_te = patient_split(pid)
    Xtr, ytr, Xte, yte = X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]
    print(f"segments={len(y)} train={len(ytr)} test={len(yte)} patients_train={len(set(pid[idx_tr]))} "
          f"patients_test={len(set(pid[idx_te]))}", flush=True)
    print("class counts train:", torch.bincount(ytr, minlength=4).tolist(), flush=True)
    parts = dirichlet_partition(ytr, args.clients, args.alpha)
    print("client shard sizes:", [len(p) for p in parts], flush=True)

    results = {}
    with open(OUT_DIR / "respiratory_curves.csv", "w") as log:
        log.write("method,round,accuracy,macro_f1,loss\n")
        results["FedAvg (no DP)"] = run("FedAvg (no DP)", None, args, Xtr, ytr, Xte, yte, parts, log)
        for eps in [float(e) for e in args.epsilons.split(",")]:
            tag = f"FedAvg+DP eps={eps}"
            results[tag] = run(tag, eps, args, Xtr, ytr, Xte, yte, parts, log)

    with open(OUT_DIR / "respiratory_summary.json", "w") as f:
        json.dump({"config": vars(args), "train": len(ytr), "test": len(yte),
                   "final": {k: {"accuracy": v[0], "macro_f1": v[1], "loss": v[2]}
                             for k, v in results.items()}}, f, indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

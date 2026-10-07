import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import get_model

app = FastAPI(title="Privacy-Preserving Federated Symptom Checker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"

SKIN_LESION_LIST = ["Actinic keratoses (akiec)", "Basal cell carcinoma (bcc)", "Benign keratosis (bkl)", "Dermatofibroma (df)", "Melanoma (mel)", "Melanocytic nevi (nv)", "Vascular lesions (vasc)"]
RESPIRATORY_COND_LIST = ["Normal", "Crackle detected", "Wheeze detected", "Both Crackle & Wheeze"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Global state populated at startup
models: Dict[str, torch.nn.Module] = {}
model_trained: Dict[str, bool] = {}
symptom_vocab: List[str] = []
disease_classes: List[str] = []
symptom_metrics: Dict[str, Any] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_symptom_mlp():
    """Loads the SymptomMLP checkpoint together with the exact feature/class
    ordering it was trained with. Ordering must come from the training run,
    not be hardcoded here, or predictions silently point at the wrong class."""
    vocab_path = CKPT_DIR / "symptom_mlp_vocab.json"
    classes_path = CKPT_DIR / "symptom_mlp_classes.json"
    ckpt_path = CKPT_DIR / "symptom_mlp_latest.npy"
    metrics_path = CKPT_DIR / "symptom_mlp_metrics.json"

    if not (vocab_path.exists() and classes_path.exists() and ckpt_path.exists()):
        print("WARNING: symptom_mlp checkpoint/vocab/classes not found. "
              "Run `python benchmarks/train_and_export_checkpoint.py` to train it. "
              "The /predict/symptoms endpoint will return 503 until then.")
        return None, [], [], {}

    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(classes_path) as f:
        classes = json.load(f)
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    model = get_model('symptom_mlp', input_dim=len(vocab), num_classes=len(classes)).to(device)
    params = np.load(ckpt_path, allow_pickle=True)
    state_dict = {k: torch.tensor(p) for k, p in zip(model.state_dict().keys(), params)}
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab, classes, metrics


def load_untrained(model_name: str) -> torch.nn.Module:
    """skin_cnn / respiratory_cnn have no real training data available
    (HAM10000 / ICBHI require Kaggle credentials this deployment doesn't
    have). Loaded with random-initialized heads so the endpoints stay
    live for UI/pipeline demonstration, but callers must be told the
    result isn't a trained prediction (see `trained` flag in responses)."""
    model = get_model(model_name).to(device)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    global models, model_trained, symptom_vocab, disease_classes, symptom_metrics
    print("Starting API Server and loading models...")

    symptom_model, vocab, classes, metrics = load_symptom_mlp()
    symptom_vocab = vocab
    disease_classes = classes
    symptom_metrics = metrics
    if symptom_model is not None:
        models['symptom_mlp'] = symptom_model
        model_trained['symptom_mlp'] = True
    else:
        model_trained['symptom_mlp'] = False

    models['skin_cnn'] = load_untrained('skin_cnn')
    model_trained['skin_cnn'] = False
    models['respiratory_cnn'] = load_untrained('respiratory_cnn')
    model_trained['respiratory_cnn'] = False


class SymptomRequest(BaseModel):
    symptoms: List[str]


@app.post("/predict/symptoms")
async def predict_symptoms(request: SymptomRequest) -> Dict[str, Any]:
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms list cannot be empty")

    if not model_trained.get('symptom_mlp'):
        raise HTTPException(status_code=503, detail="Symptom model not trained yet. Run "
                             "benchmarks/train_and_export_checkpoint.py on the server.")

    unknown = [s for s in request.symptoms if s not in symptom_vocab]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unrecognized symptom(s): {unknown}")

    features = torch.zeros((1, len(symptom_vocab)), device=device)
    for sym in request.symptoms:
        features[0, symptom_vocab.index(sym)] = 1.0

    model = models['symptom_mlp']
    with torch.no_grad():
        output = model(features)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()

    top_pred_idx = int(np.argmax(probs))

    top_predictions = [
        {
            "disease": disease_classes[i],
            "confidence": float(probs[i]),
            "probability": float(probs[i])
        }
        for i in range(len(probs))
    ]
    top_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "disease": disease_classes[top_pred_idx],
        "confidence": float(probs[top_pred_idx]),
        "probability": float(probs[top_pred_idx]),
        "top_predictions": top_predictions[:5],
        "trained": True,
        "benchmark_metrics": symptom_metrics,
    }


def preprocess_skin_image(content: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.content_type or "").startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    content = await file.read()
    try:
        input_tensor = preprocess_skin_image(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    model = models['skin_cnn']
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()

    top_pred_idx = int(np.argmax(probs))

    top_predictions = [
        {"disease": SKIN_LESION_LIST[i], "confidence": float(probs[i])}
        for i in range(len(probs))
    ]
    top_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "lesion_type": SKIN_LESION_LIST[top_pred_idx],
        "confidence": float(probs[top_pred_idx]),
        "top_predictions": top_predictions[:5],
        "trained": False,
        "notice": "This classifier's head has not been trained on real skin-lesion data "
                  "(no HAM10000 access in this deployment) -- result is illustrative only."
    }


def preprocess_respiratory_audio(content: bytes) -> torch.Tensor:
    import librosa
    y, sr = librosa.load(io.BytesIO(content), sr=22050, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Resize time axis to 128 frames (pad or truncate)
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :128]
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return torch.tensor(mel_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


@app.post("/predict/respiratory")
async def predict_respiratory(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.content_type or "").startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload audio.")

    content = await file.read()
    try:
        input_tensor = preprocess_respiratory_audio(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read/decode audio file.")

    model = models['respiratory_cnn']
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()

    top_pred_idx = int(np.argmax(probs))

    top_predictions = [
        {"disease": RESPIRATORY_COND_LIST[i], "confidence": float(probs[i])}
        for i in range(len(probs))
    ]
    top_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "condition": RESPIRATORY_COND_LIST[top_pred_idx],
        "confidence": float(probs[top_pred_idx]),
        "top_predictions": top_predictions[:5],
        "trained": False,
        "notice": "This classifier has not been trained on real respiratory-sound data "
                  "(no ICBHI 2017 access in this deployment) -- result is illustrative only."
    }


@app.get("/model/status")
async def get_model_status() -> Dict[str, Any]:
    if model_trained.get('symptom_mlp'):
        return {
            "current_round": symptom_metrics.get("rounds", 0),
            "total_rounds": symptom_metrics.get("rounds", 0),
            "global_accuracy": symptom_metrics.get("accuracy", 0.0),
            "num_clients": symptom_metrics.get("num_clients", 0),
            "method": symptom_metrics.get("method", ""),
        }
    return {
        "current_round": 0,
        "total_rounds": 0,
        "global_accuracy": None,
        "num_clients": 0,
        "method": "not trained",
    }


@app.get("/privacy/budget")
async def get_privacy_budget() -> Dict[str, Any]:
    # Reference configuration benchmarked in benchmarks/real_comparison_experiment.py
    # (see benchmarks/results/comparison_results.csv). The deployed symptom_mlp
    # checkpoint itself is FedAvg WITHOUT differential privacy (see /model/status);
    # DP variants at this project's benchmarked round budget cut accuracy sharply,
    # so DP is not applied to the live inference model. These figures describe the
    # benchmark, not a guarantee active on the deployed model.
    return {
        "epsilon": 2.0,
        "delta": 1e-5,
        "note": "Reference privacy-utility benchmark (moderate preset, eps=2.0). "
                "The deployed symptom checker uses FedAvg without DP for usable "
                "accuracy; see benchmarks/results/comparison_results.csv for the "
                "measured accuracy cost at eps in {0.5, 1.0, 2.0, 5.0}.",
        "dp_applied_to_deployed_model": False,
    }


@app.get("/symptoms")
async def get_symptoms() -> Dict[str, Any]:
    return {"symptoms": symptom_vocab, "trained": model_trained.get('symptom_mlp', False)}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api_server:app", host="0.0.0.0", port=8000, reload=True)

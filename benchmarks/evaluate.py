import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from typing import Dict, Any, Tuple

from models import get_model

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str, num_classes: int) -> Dict[str, Any]:
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = float(accuracy_score(all_targets, all_preds))
    precision, recall, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    _, _, f1_per_class, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    report = classification_report(all_targets, all_preds, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_macro),
        'per_class_f1': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def measure_inference_latency(model: nn.Module, input_shape: Tuple[int, ...], device: str, num_runs: int = 100) -> Dict[str, float]:
    """Measure inference latency of the model."""
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    if device == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
            
    return {
        'mean_latency_ms': float(np.mean(times)),
        'std_latency_ms': float(np.std(times)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'p99_latency_ms': float(np.percentile(times, 99))
    }

def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }

def evaluate_communication_cost(model: nn.Module, num_clients: int, num_rounds: int) -> Dict[str, float]:
    """Calculate FL communication cost."""
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_model = total_params * 4
    bytes_per_round = bytes_per_model * num_clients * 2  # upload + download
    total_bytes = bytes_per_round * num_rounds
    
    return {
        'bytes_per_round': float(bytes_per_round),
        'total_bytes': float(total_bytes),
        'total_mb': float(total_bytes / (1024 ** 2))
    }

def generate_evaluation_report(metrics_dict: Dict[str, Any], output_path: str) -> None:
    """Save text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("="*50)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*50 + "\n")
    
    if 'accuracy' in metrics_dict:
        report.append("--- Classification Metrics ---")
        report.append(f"Accuracy:  {metrics_dict['accuracy']:.4f}")
        report.append(f"Precision: {metrics_dict['precision']:.4f}")
        report.append(f"Recall:    {metrics_dict['recall']:.4f}")
        report.append(f"F1 Score:  {metrics_dict['f1_score']:.4f}")
        report.append("\nClassification Report:")
        report.append(metrics_dict['classification_report'])
        report.append("\n")
        
    if 'mean_latency_ms' in metrics_dict:
        report.append("--- Inference Latency ---")
        report.append(f"Mean: {metrics_dict['mean_latency_ms']:.2f} ms")
        report.append(f"Std:  {metrics_dict['std_latency_ms']:.2f} ms")
        report.append(f"p95:  {metrics_dict['p95_latency_ms']:.2f} ms")
        report.append(f"p99:  {metrics_dict['p99_latency_ms']:.2f} ms")
        report.append("\n")
        
    if 'total_params' in metrics_dict:
        report.append("--- Model Parameters ---")
        report.append(f"Total Params:     {metrics_dict['total_params']:,}")
        report.append(f"Trainable Params: {metrics_dict['trainable_params']:,}")
        report.append(f"Size:             {metrics_dict['model_size_mb']:.2f} MB")
        report.append("\n")
        
    if 'total_mb' in metrics_dict:
        report.append("--- Communication Cost (FL) ---")
        report.append(f"Bytes per round: {metrics_dict['bytes_per_round']:,.0f}")
        report.append(f"Total MB:        {metrics_dict['total_mb']:.2f} MB")
        report.append("\n")
        
    with open(output_path, 'w') as f:
        f.write("\n".join(report))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument('--model', type=str, required=True, help="Model name ('symptom_mlp', 'skin_cnn', 'respiratory_cnn')")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model checkpoint")
    parser.add_argument('--dataset', type=str, default=None, help="Path to dataset")
    parser.add_argument('--device', type=str, default='cpu', help="Device (cpu/cuda)")
    parser.add_argument('--output_dir', type=str, default='results', help="Output directory")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Instantiate model
    model = get_model(args.model)
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    
    input_shape = (1, 132) if args.model == 'symptom_mlp' else ((1, 3, 224, 224) if args.model == 'skin_cnn' else (1, 1, 128, 128))
    
    metrics = {}
    metrics.update(count_parameters(model))
    metrics.update(measure_inference_latency(model, input_shape, device))
    metrics.update(evaluate_communication_cost(model, num_clients=10, num_rounds=100))
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_evaluation_report(metrics, str(out_dir / f"{args.model}_eval_report.txt"))
    print(f"Evaluation report saved to {out_dir / f'{args.model}_eval_report.txt'}")

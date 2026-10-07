import argparse
import time
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Tuple

from models import get_model

def export_torchscript(model: nn.Module, input_shape: Tuple[int, ...], output_path: str) -> None:
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Verify
        orig_out = model(dummy_input)
        traced_out = traced_model(dummy_input)
        assert torch.allclose(orig_out, traced_out, atol=1e-4), "TorchScript verification failed!"
        
    traced_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")

def export_onnx(model: nn.Module, input_shape: Tuple[int, ...], output_path: str) -> None:
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {output_path}")

def quantize_model(model: nn.Module, input_shape: Tuple[int, ...]) -> nn.Module:
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration with dummy data
    for _ in range(10):
        dummy_input = torch.randn(*input_shape)
        model(dummy_input)
        
    torch.quantization.convert(model, inplace=True)
    return model

def benchmark_export(original_model: nn.Module, exported_path: str, input_shape: Tuple[int, ...], device: str) -> None:
    original_model.eval()
    original_model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Latency Original
    start = time.perf_counter()
    for _ in range(100):
        _ = original_model(dummy_input)
    orig_time = (time.perf_counter() - start) / 100 * 1000
    
    # Check exported file size
    size_mb = os.path.getsize(exported_path) / (1024 ** 2) if os.path.exists(exported_path) else 0.0
    
    print(f"Original model latency: {orig_time:.2f} ms")
    print(f"Exported model size: {size_mb:.2f} MB")
    
def export_all_formats(model_name: str, checkpoint_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = get_model(model_name)
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
    input_shape = (1, 132) if model_name == 'symptom_mlp' else ((1, 3, 224, 224) if model_name == 'skin_cnn' else (1, 1, 128, 128))
    
    ts_path = output_dir / f"{model_name}.pt"
    export_torchscript(model, input_shape, str(ts_path))
    
    onnx_path = output_dir / f"{model_name}.onnx"
    export_onnx(model, input_shape, str(onnx_path))
    
    # We load fresh model for quantization
    model_q = get_model(model_name)
    if checkpoint_path and Path(checkpoint_path).exists():
        model_q.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    try:
        q_model = quantize_model(model_q, input_shape)
        ts_q_path = output_dir / f"{model_name}_quantized.pt"
        
        dummy_input = torch.randn(*input_shape)
        traced_q = torch.jit.trace(q_model, dummy_input)
        traced_q.save(str(ts_q_path))
        print(f"Quantized TorchScript model saved to {ts_q_path}")
    except Exception as e:
        print(f"Quantization failed (might be unsupported ops): {e}")
        
    print("\nExport complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument('--model', type=str, required=True, help="Model name")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint")
    parser.add_argument('--output_dir', type=str, default='exports', help="Output directory")
    args = parser.parse_args()
    
    ckpt = args.checkpoint if args.checkpoint else f"checkpoints/{args.model}_best.pt"
    export_all_formats(args.model, ckpt, args.output_dir)

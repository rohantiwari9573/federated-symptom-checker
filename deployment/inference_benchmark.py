import argparse
import time
import os
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from models import get_model

def benchmark_model(model: Any, input_shape: Tuple[int, ...], device: str, num_iterations: int = 1000) -> Dict[str, float]:
    dummy_input = torch.randn(*input_shape)
    if hasattr(model, 'to'):
        model.to(device)
        dummy_input = dummy_input.to(device)
        
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    if device == 'cuda':
        torch.cuda.synchronize()
        
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        
    mean = np.mean(times)
    return {
        'mean_latency_ms': float(mean),
        'std_latency_ms': float(np.std(times)),
        'p50_latency_ms': float(np.percentile(times, 50)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'p99_latency_ms': float(np.percentile(times, 99)),
        'throughput_fps': float(1000.0 / mean) if mean > 0 else 0.0
    }

def measure_memory_footprint(model: Any, input_shape: Tuple[int, ...], device: str) -> float:
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randn(*input_shape).to(device)
        _ = model(dummy_input)
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        dummy_input = torch.randn(*input_shape)
        _ = model(dummy_input)
        mem_after = process.memory_info().rss
        return (mem_after - mem_before) / (1024 ** 2)

def benchmark_all_formats(model_name: str, exports_dir: str, device: str, iterations: int = 1000) -> None:
    input_shape = (1, 132) if model_name == 'symptom_mlp' else ((1, 3, 224, 224) if model_name == 'skin_cnn' else (1, 1, 128, 128))
    exports_dir = Path(exports_dir)
    
    results = {}
    
    # 1. PyTorch FP32
    print("Benchmarking PyTorch FP32...")
    model = get_model(model_name)
    model.eval()
    results['PyTorch FP32'] = benchmark_model(model, input_shape, device, iterations)
    
    # 2. TorchScript
    ts_path = exports_dir / f"{model_name}.pt"
    if ts_path.exists():
        print("Benchmarking TorchScript...")
        ts_model = torch.jit.load(str(ts_path))
        results['TorchScript'] = benchmark_model(ts_model, input_shape, device, iterations)
        
    # 3. ONNX Runtime
    onnx_path = exports_dir / f"{model_name}.onnx"
    if onnx_path.exists():
        print("Benchmarking ONNX Runtime...")
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Custom benchmark loop for ONNX
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            for _ in range(10):
                _ = ort_session.run(None, {'input': dummy_input})
                
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = ort_session.run(None, {'input': dummy_input})
                end = time.perf_counter()
                times.append((end - start) * 1000)
                
            mean = np.mean(times)
            results['ONNX Runtime'] = {
                'mean_latency_ms': float(mean),
                'std_latency_ms': float(np.std(times)),
                'p50_latency_ms': float(np.percentile(times, 50)),
                'p95_latency_ms': float(np.percentile(times, 95)),
                'p99_latency_ms': float(np.percentile(times, 99)),
                'throughput_fps': float(1000.0 / mean) if mean > 0 else 0.0
            }
        except ImportError:
            print("onnxruntime not installed, skipping ONNX benchmark.")
            
    # 4. Quantized INT8
    q_path = exports_dir / f"{model_name}_quantized.pt"
    if q_path.exists() and device == 'cpu':
        print("Benchmarking Quantized INT8...")
        q_model = torch.jit.load(str(q_path))
        results['Quantized INT8'] = benchmark_model(q_model, input_shape, 'cpu', iterations)
        
    print("\n" + "="*80)
    print(f"INFERENCE BENCHMARK RESULTS ({model_name} on {device})")
    print("="*80)
    print(f"{'Format':<20} | {'Mean (ms)':<10} | {'p95 (ms)':<10} | {'Throughput':<12}")
    print("-" * 80)
    for fmt, stats in results.items():
        print(f"{fmt:<20} | {stats['mean_latency_ms']:<10.2f} | {stats['p95_latency_ms']:<10.2f} | {stats['throughput_fps']:<10.2f} fps")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument('--model', type=str, required=True, help="Model name")
    parser.add_argument('--exports_dir', type=str, default='exports', help="Directory with exported models")
    parser.add_argument('--device', type=str, default='cpu', help="Device (cpu/cuda)")
    parser.add_argument('--iterations', type=int, default=1000, help="Number of iterations")
    args = parser.parse_args()
    
    benchmark_all_formats(args.model, args.exports_dir, args.device, args.iterations)

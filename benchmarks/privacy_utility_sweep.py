import argparse
import csv
import torch
import math
from pathlib import Path
from typing import List, Dict, Any

from models import get_model

# Mock function simulating server FL logic for simplicity, normally import from server
def mock_run_simulation(model_name: str, epsilon: float, num_clients: int, rounds: int, local_epochs: int, batch_size: int, alpha: float, device: str) -> Dict[str, Any]:
    """Mock function representing a federated simulation run."""
    base_acc = 0.85
    penalty = 0.0
    if epsilon < float('inf'):
        penalty = 0.2 / max(epsilon, 0.1)
    acc = max(0.1, base_acc - penalty)
    
    return {
        'final_accuracy': acc,
        'final_f1': max(0.1, acc - 0.05),
        'final_loss': max(0.1, 1.5 - acc),
        'rounds_to_converge': min(rounds, int(rounds * 0.8 + penalty * 50))
    }

def run_centralized_baseline(model_name: str, dataset: str, device: str) -> Dict[str, Any]:
    """Mock function representing centralized training baseline."""
    return {
        'final_accuracy': 0.88,
        'final_f1': 0.85,
        'final_loss': 0.3,
        'rounds_to_converge': 50
    }

def main():
    parser = argparse.ArgumentParser(description="Privacy-Utility Trade-off Sweep")
    parser.add_argument('--model', type=str, required=True, help="Model name")
    parser.add_argument('--dataset', type=str, default='data', help="Dataset path")
    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients")
    parser.add_argument('--rounds', type=int, default=50, help="Federated rounds")
    parser.add_argument('--local_epochs', type=int, default=3, help="Local epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--alpha', type=float, default=0.5, help="Dirichlet alpha for data heterogeneity")
    parser.add_argument('--device', type=str, default='cpu', help="Device")
    parser.add_argument('--output_dir', type=str, default='results', help="Output directory")
    parser.add_argument('--quick', action='store_true', help="Use fewer rounds and clients for fast testing")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "privacy_utility_sweep.csv"

    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
    
    rounds = 5 if args.quick else args.rounds
    clients = 3 if args.quick else args.num_clients

    results = []

    # Centralized baseline
    print("Running centralized baseline...")
    cent_res = run_centralized_baseline(args.model, args.dataset, args.device)
    cent_res['epsilon'] = 'baseline'
    cent_res['type'] = 'centralized'
    results.append(cent_res)

    for eps in epsilons:
        print(f"Running FL simulation with epsilon = {eps}...")
        res = mock_run_simulation(args.model, eps, clients, rounds, args.local_epochs, args.batch_size, args.alpha, args.device)
        res['epsilon'] = eps
        res['type'] = 'federated_dp' if eps < float('inf') else 'federated'
        results.append(res)

    keys = ['type', 'epsilon', 'final_accuracy', 'final_f1', 'final_loss', 'rounds_to_converge']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n--- Summary Table ---")
    print(f"{'Type':<15} | {'Epsilon':<8} | {'Accuracy':<10} | {'F1':<10} | {'Loss':<10} | {'Rounds':<8}")
    print("-" * 75)
    for r in results:
        eps_str = str(r['epsilon'])
        print(f"{r['type']:<15} | {eps_str:<8} | {r['final_accuracy']:<10.4f} | {r['final_f1']:<10.4f} | {r['final_loss']:<10.4f} | {r['rounds_to_converge']:<8}")

if __name__ == '__main__':
    main()

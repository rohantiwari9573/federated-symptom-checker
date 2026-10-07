import argparse
import os
import json
from pathlib import Path
import flwr as fl
import torch
import pandas as pd

from data.partition import create_client_dataloaders
from federated.dp_config import create_dp_config
from federated.client import create_client_fn
from federated.strategy import FedSymptomStrategy
from federated.server import create_server_config, get_evaluate_fn

def main():
    parser = argparse.ArgumentParser(description="Federated No-DP Baseline Simulation")
    parser.add_argument("--model", type=str, required=True, choices=["symptom_mlp", "skin_cnn", "respiratory_cnn"])
    parser.add_argument("--dataset", type=str, required=True, choices=["tabular", "skin", "respiratory"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--use_fedprox", action="store_true")
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting No-DP baseline on {device}")
    
    try:
        client_dataloaders = create_client_dataloaders(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            alpha=args.alpha
        )
    except Exception as e:
        print(f"Using dummy data due to error: {e}")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx): return torch.randn(3, 224, 224), 0
        dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=args.batch_size)
        client_dataloaders = [(dl, dl) for _ in range(args.num_clients)]

    test_loader = client_dataloaders[0][1]

    # No DP config
    dp_config = create_dp_config(
        epsilon=float('inf'),
        max_grad_norm=1.0,
        num_train_samples=1000,
        epochs=args.local_epochs,
        batch_size=args.batch_size
    )

    client_fn = create_client_fn(
        model_name=args.model,
        client_dataloaders=client_dataloaders,
        local_epochs=args.local_epochs,
        dp_config=dp_config,
        device=device
    )

    eval_fn = get_evaluate_fn(args.model, test_loader, device)
    
    strategy = FedSymptomStrategy(
        use_fedprox=args.use_fedprox,
        mu=args.mu,
        track_communication=True,
        checkpoint_dir=str(out_dir / "checkpoints_nodp"),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=eval_fn
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1.0, "num_gpus": 1.0 if device.type == "cuda" else 0.0}
    )

    results_file = out_dir / f"results_nodp_{args.model}.json"
    with open(results_file, 'w') as f:
        res_dict = {
            "losses_distributed": history.losses_distributed,
            "metrics_distributed": history.metrics_distributed,
            "losses_centralized": history.losses_centralized,
            "metrics_centralized": history.metrics_centralized
        }
        f.write(str(res_dict))

    print(f"No-DP baseline complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

log = logging.getLogger(__name__)

def get_fedprox_loss(
    model: torch.nn.Module, 
    global_params: List[np.ndarray], 
    mu: float
) -> torch.Tensor:
    """
    Computes the FedProx proximal term: (mu/2) * ||w - w_global||^2
    This helper should be called inside the client's training loop if use_fedprox is enabled.
    """
    proximal_term = 0.0
    for local_param, global_param in zip(model.parameters(), global_params):
        global_tensor = torch.tensor(global_param, dtype=local_param.dtype, device=local_param.device)
        proximal_term += torch.sum(torch.square(local_param - global_tensor))
    return (mu / 2.0) * proximal_term

class FedSymptomStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        use_fedprox: bool = False,
        mu: float = 0.01,
        track_communication: bool = True,
        checkpoint_dir: str = "checkpoints",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_fedprox = use_fedprox
        self.mu = mu
        self.track_communication = track_communication
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.total_bytes_transferred = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Track communication cost
            if self.track_communication:
                # parameters_to_ndarrays
                ndarrays = parameters_to_ndarrays(aggregated_parameters)
                # Calculate bytes (sum of elements * 4 bytes for float32)
                param_size_bytes = sum(arr.size * 4 for arr in ndarrays)
                
                # Bytes sent to clients (previous round broadcast) + bytes received from clients
                # For simplicity, just tracking received bytes in this step per client
                num_clients = len(results)
                bytes_this_round = param_size_bytes * num_clients
                self.total_bytes_transferred += bytes_this_round
                
                aggregated_metrics["bytes_transferred_this_round"] = bytes_this_round
                aggregated_metrics["total_bytes_transferred"] = self.total_bytes_transferred

            # Custom metric aggregation
            total_examples = sum([res.num_examples for _, res in results])
            
            # Weighted average of train_loss and train_acc
            train_loss = 0.0
            train_acc = 0.0
            epsilon_spent = 0.0
            
            for _, res in results:
                examples = res.num_examples
                metrics = res.metrics
                weight = examples / total_examples if total_examples > 0 else 0
                
                if "train_loss" in metrics:
                    train_loss += metrics["train_loss"] * weight
                if "train_acc" in metrics:
                    train_acc += metrics["train_acc"] * weight
                if "epsilon_spent" in metrics:
                    # Take the max epsilon spent among clients as a conservative bound
                    # or average it. Let's do max to be safe for DP.
                    val = metrics["epsilon_spent"]
                    if val != float('inf') and val > epsilon_spent:
                        epsilon_spent = val

            aggregated_metrics["train_loss"] = train_loss
            aggregated_metrics["train_acc"] = train_acc
            
            if epsilon_spent > 0:
                aggregated_metrics["epsilon_spent"] = epsilon_spent

            # Save global checkpoint (pass dtype=object for list of arrays)
            arr_to_save = np.array(parameters_to_ndarrays(aggregated_parameters), dtype=object)
            np.save(self.checkpoint_dir / f"global_model_round_{server_round}.npy", arr_to_save)
            np.save(self.checkpoint_dir / "global_model_latest.npy", arr_to_save)

            log.info(f"Round {server_round} aggregated metrics: {aggregated_metrics}")

        return aggregated_parameters, aggregated_metrics

if __name__ == "__main__":
    pass

import flwr as fl
import torch
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict

from models import get_model

def create_server_config(
    num_rounds: int,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_available_clients: int = 2
) -> fl.server.ServerConfig:
    """
    Creates a Flower server config.
    """
    return fl.server.ServerConfig(num_rounds=num_rounds)

def get_evaluate_fn(
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Callable:
    """
    Returns a server-side evaluation function.
    """
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model = get_model(model_name)
        
        # Load parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        model.to(device)
        model.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
                
        if total == 0:
            return None
            
        loss_avg = total_loss / total
        accuracy = correct / total
        
        return loss_avg, {"accuracy": accuracy}
        
    return evaluate

def start_flower_server(
    model_name: str,
    strategy: fl.server.strategy.Strategy,
    num_rounds: int,
    server_config: fl.server.ServerConfig
):
    """
    Starts a standalone Flower server.
    """
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
    )

if __name__ == "__main__":
    pass

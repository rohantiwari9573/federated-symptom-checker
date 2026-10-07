import copy
import warnings
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from opacus import PrivacyEngine

from models import get_model
from federated.dp_config import DPConfig, make_private

warnings.filterwarnings("ignore")

class FedSymptomClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        local_epochs: int,
        dp_config: DPConfig,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.dp_config = dp_config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        # Unwrap model if it was wrapped by Opacus GradSampleModule
        model_to_extract = self.model._module if hasattr(self.model, '_module') else self.model
        return [val.cpu().numpy() for _, val in model_to_extract.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        model_to_load = self.model._module if hasattr(self.model, '_module') else self.model
        params_dict = zip(model_to_load.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model_to_load.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        self.set_parameters(parameters)
        
        # Deep copy the model to avoid state leakage and Opacus multi-wrap issues
        # Or we can just use the provided model
        local_model = copy.deepcopy(self.model._module if hasattr(self.model, '_module') else self.model)
        local_model.train()
        local_model.to(self.device)

        if self.dp_config.enabled:
            from opacus.validators import ModuleValidator
            if not ModuleValidator.is_valid(local_model):
                local_model = ModuleValidator.fix(local_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
            actual_dp_config = copy.deepcopy(self.dp_config)
            local_model, optimizer, train_loader, privacy_engine = make_private(
                local_model, optimizer, self.train_loader, actual_dp_config
            )
        else:
            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
            train_loader = self.train_loader
            privacy_engine = None

        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        # Update original model with trained parameters
        self.model = local_model

        metrics: Dict[str, fl.common.Scalar] = {
            "train_loss": total_loss / total if total > 0 else 0.0,
            "train_acc": correct / total if total > 0 else 0.0,
        }

        if privacy_engine is not None:
            epsilon_spent = privacy_engine.get_epsilon(self.dp_config.target_delta)
            metrics["epsilon_spent"] = epsilon_spent
        else:
            metrics["epsilon_spent"] = float('inf')

        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        self.set_parameters(parameters)
        
        eval_model = self.model._module if hasattr(self.model, '_module') else self.model
        eval_model.eval()
        eval_model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        loss_avg = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return float(loss_avg), total, {"val_acc": accuracy}

def create_client_fn(
    model_name: str,
    client_dataloaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    local_epochs: int,
    dp_config: DPConfig,
    device: torch.device
) -> Callable[[str], fl.client.Client]:
    """
    Returns a function that creates a FedSymptomClient for a given client_id.
    """
    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        train_loader, val_loader = client_dataloaders[cid_int]
        
        # Instantiate a fresh model for the client
        model = get_model(model_name)
        
        client = FedSymptomClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=local_epochs,
            dp_config=dp_config,
            device=device
        )
        return client.to_client()
    return client_fn

if __name__ == "__main__":
    # Example test
    pass

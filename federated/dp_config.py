import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator

@dataclass
class DPConfig:
    target_epsilon: float
    target_delta: float
    max_grad_norm: float
    noise_multiplier: float
    enabled: bool

PRIVACY_PRESETS = {
    'strict': {'epsilon': 0.5},
    'moderate': {'epsilon': 2.0},
    'relaxed': {'epsilon': 5.0},
    'benchmark': {'epsilon': 10.0},
    'none': {'epsilon': float('inf')}
}

def create_dp_config(
    epsilon: float,
    delta: Optional[float] = None,
    max_grad_norm: float = 1.0,
    num_train_samples: int = 1000,
    epochs: int = 1,
    batch_size: int = 32
) -> DPConfig:
    """
    Creates a DP configuration with computed noise multiplier.
    """
    if epsilon == float('inf'):
        return DPConfig(
            target_epsilon=float('inf'),
            target_delta=0.0,
            max_grad_norm=max_grad_norm,
            noise_multiplier=0.0,
            enabled=False
        )

    if delta is None:
        delta = 1.0 / max(100, num_train_samples)

    sample_rate = batch_size / num_train_samples

    # Compute noise multiplier using Opacus utils
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp"
        )
    except Exception as e:
        print(f"Warning: Could not compute precise noise multiplier ({e}). Using approximation.")
        # Very rough approximation if accountant fails
        noise_multiplier = (math.sqrt(2 * math.log(1.25 / delta)) / epsilon) * math.sqrt(epochs)

    return DPConfig(
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        enabled=True
    )

def make_private(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp_config: DPConfig
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, Optional[PrivacyEngine]]:
    """
    Wraps model, optimizer, and data_loader with Opacus PrivacyEngine.
    Also handles module compatibility like BatchNorm replacement.
    """
    if not dp_config.enabled:
        return model, optimizer, data_loader, None

    # Replace unsupported modules (e.g., BatchNorm -> GroupNorm)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
    
    # We must be careful not to wrap multiple times in FL loops, 
    # but PrivacyEngine handles it typically if we do it per round properly, 
    # or by re-initializing. Here we assume fresh wrap.
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
    )

    return model, optimizer, data_loader, privacy_engine

if __name__ == '__main__':
    # Example usage
    config = create_dp_config(epsilon=2.0, num_train_samples=5000, epochs=3, batch_size=32)
    print(f"Computed noise multiplier: {config.noise_multiplier:.4f}")

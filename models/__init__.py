"""
Model registry and factory functions.
Supports: 'symptom_mlp', 'skin_cnn', 'respiratory_cnn'
"""

from typing import Dict, Any, Type
import torch.nn as nn
from .symptom_mlp import SymptomMLP
from .skin_cnn import SkinCNN
from .respiratory_cnn import RespiratoryCNN

__all__ = ['get_model', 'get_model_info', 'SymptomMLP', 'SkinCNN', 'RespiratoryCNN']

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'symptom_mlp': SymptomMLP,
    'skin_cnn': SkinCNN,
    'respiratory_cnn': RespiratoryCNN
}

def get_model(name: str, **kwargs) -> nn.Module:
    """Factory function to get a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

def get_model_info(name: str) -> Dict[str, Any]:
    """Return model parameter count, input shape, and output classes."""
    model = get_model(name)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'parameters': params,
    }
    
    if name == 'symptom_mlp':
        info['input_shape'] = (1, 132)
        info['output_classes'] = 41
    elif name == 'skin_cnn':
        info['input_shape'] = (1, 3, 224, 224)
        info['output_classes'] = 7
    elif name == 'respiratory_cnn':
        info['input_shape'] = (1, 1, 128, 128)
        info['output_classes'] = 4
        
    return info

"""
Tabular symptom classifier.
~50K parameters, lightweight MLP.
"""

import torch
import torch.nn as nn

class SymptomMLP(nn.Module):
    """
    Multilayer Perceptron for tabular symptom features.
    Input: binary symptom vector (132 features)
    Output: 41 disease classes
    """
    def __init__(self, input_dim: int = 131, num_classes: int = 41):
        super(SymptomMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    model = SymptomMLP()
    x = torch.randn(16, 132) # Batch size 16
    out = model(x)
    print(f"SymptomMLP Output shape: {out.shape} (Expected: 16, 41)")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

"""
Respiratory sound classifier using custom CNN for Mel spectrograms.
"""

import torch
import torch.nn as nn

class RespiratoryCNN(nn.Module):
    """
    CNN for Respiratory Sound Classification.
    Input: 1x128x128 Mel spectrogram
    Output: 4 classes (Normal, Crackle, Wheeze, Both)
    """
    def __init__(self, num_classes: int = 4):
        super(RespiratoryCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 1 -> 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = RespiratoryCNN()
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    print(f"RespiratoryCNN Output shape: {out.shape} (Expected: 4, 4)")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

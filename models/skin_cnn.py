"""
Skin lesion classifier using MobileNetV3-Small backbone.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SkinCNN(nn.Module):
    """
    CNN for Skin Lesion Classification.
    Uses MobileNetV3-Small backbone.
    Input: 224x224x3 RGB images
    Output: 7 HAM10000 classes
    """
    def __init__(self, num_classes: int = 7):
        super(SkinCNN, self).__init__()
        # Load pretrained MobileNetV3-Small
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights)
        
        # Replace classifier head
        # MobileNetV3-Small features output 576 channels before classifier
        in_features = 576
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

if __name__ == "__main__":
    model = SkinCNN()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"SkinCNN Output shape: {out.shape} (Expected: 2, 7)")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

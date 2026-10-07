"""
Test suite for ML models in the Federated Symptom Checker.

Verifies model architecture, forward pass, parameter counts,
and compatibility with Opacus for Differential Privacy.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSymptomMLP:
    """Tests for the tabular symptom classifier (MLP)."""

    def setup_method(self):
        """Set up test fixtures."""
        from models.symptom_mlp import SymptomMLP
        self.model = SymptomMLP()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_forward_pass_single(self):
        """Test forward pass with a single sample (eval mode for BatchNorm)."""
        self.model.eval()
        x = torch.randn(1, 131)
        output = self.model(x)
        assert output.shape == (1, 41), f"Expected (1, 41), got {output.shape}"

    def test_forward_pass_batch(self):
        """Test forward pass with a batch of samples."""
        x = torch.randn(32, 131)
        output = self.model(x)
        assert output.shape == (32, 41), f"Expected (32, 41), got {output.shape}"

    def test_output_not_nan(self):
        """Test that model output contains no NaN values."""
        x = torch.randn(16, 131)
        output = self.model(x)
        assert not torch.isnan(output).any(), "Model output contains NaN values"

    def test_parameter_count(self):
        """Test that model has reasonable parameter count (should be lightweight ~50K)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        assert total_params < 200_000, f"Model too large: {total_params} params"
        assert total_params > 10_000, f"Model too small: {total_params} params"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(8, 131)
        target = torch.randint(0, 41, (8,))
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_binary_input(self):
        """Test with binary input (actual use case: one-hot encoded symptoms)."""
        x = torch.zeros(4, 131)
        x[:, [0, 5, 10, 30]] = 1.0  # Select a few symptoms
        output = self.model(x)
        assert output.shape == (4, 41)


class TestSkinCNN:
    """Tests for the skin lesion classifier (MobileNetV3-based CNN)."""

    def setup_method(self):
        """Set up test fixtures."""
        from models.skin_cnn import SkinCNN
        self.model = SkinCNN()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_forward_pass_single(self):
        """Test forward pass with a single image."""
        x = torch.randn(1, 3, 224, 224)
        output = self.model(x)
        assert output.shape == (1, 7), f"Expected (1, 7), got {output.shape}"

    def test_forward_pass_batch(self):
        """Test forward pass with a batch of images."""
        x = torch.randn(4, 3, 224, 224)
        output = self.model(x)
        assert output.shape == (4, 7), f"Expected (4, 7), got {output.shape}"

    def test_output_not_nan(self):
        """Test that model output contains no NaN values."""
        x = torch.randn(2, 3, 224, 224)
        output = self.model(x)
        assert not torch.isnan(output).any(), "Model output contains NaN values"

    def test_parameter_count(self):
        """Test model parameter count (MobileNetV3-Small ~2.5M params)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        assert total_params < 5_000_000, f"Model too large: {total_params} params"
        assert total_params > 500_000, f"Model too small: {total_params} params"


class TestRespiratoryCNN:
    """Tests for the respiratory sound classifier (custom CNN)."""

    def setup_method(self):
        """Set up test fixtures."""
        from models.respiratory_cnn import RespiratoryCNN
        self.model = RespiratoryCNN()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_forward_pass_single(self):
        """Test forward pass with a single spectrogram."""
        x = torch.randn(1, 1, 128, 128)
        output = self.model(x)
        assert output.shape == (1, 4), f"Expected (1, 4), got {output.shape}"

    def test_forward_pass_batch(self):
        """Test forward pass with a batch of spectrograms."""
        x = torch.randn(8, 1, 128, 128)
        output = self.model(x)
        assert output.shape == (8, 4), f"Expected (8, 4), got {output.shape}"

    def test_output_not_nan(self):
        """Test that model output contains no NaN values."""
        x = torch.randn(4, 1, 128, 128)
        output = self.model(x)
        assert not torch.isnan(output).any(), "Model output contains NaN values"

    def test_parameter_count(self):
        """Test model parameter count (should be compact)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        assert total_params < 2_000_000, f"Model too large: {total_params} params"
        assert total_params > 50_000, f"Model too small: {total_params} params"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        x = torch.randn(4, 1, 128, 128)
        target = torch.randint(0, 4, (4,))
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelRegistry:
    """Tests for the model registry/factory."""

    def test_get_symptom_mlp(self):
        """Test creating symptom_mlp via registry."""
        from models import get_model
        model = get_model("symptom_mlp")
        assert isinstance(model, nn.Module)

    def test_get_skin_cnn(self):
        """Test creating skin_cnn via registry."""
        from models import get_model
        model = get_model("skin_cnn")
        assert isinstance(model, nn.Module)

    def test_get_respiratory_cnn(self):
        """Test creating respiratory_cnn via registry."""
        from models import get_model
        model = get_model("respiratory_cnn")
        assert isinstance(model, nn.Module)

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        from models import get_model
        with pytest.raises((ValueError, KeyError)):
            get_model("nonexistent_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

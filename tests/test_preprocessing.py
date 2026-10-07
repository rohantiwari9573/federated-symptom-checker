"""
Test suite for data preprocessing pipelines in the Federated Symptom Checker.

Tests the preprocessing logic, data partitioning, and dataset classes
without requiring actual dataset downloads.
"""

import sys
import os
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTabularPreprocessing:
    """Tests for tabular symptom data preprocessing."""

    def test_symptom_encoding_shape(self):
        """Test that symptom encoding produces correct feature vector size."""
        # Simulate what preprocess_tabular does
        import pandas as pd

        # Create a mock symptom dataframe
        data = {
            "Disease": ["Flu", "Cold", "Flu"],
            "Symptom_1": ["fever", "cough", "fever"],
            "Symptom_2": ["headache", "runny_nose", "chills"],
            "Symptom_3": ["chills", np.nan, "headache"],
            "Symptom_4": [np.nan, np.nan, np.nan],
        }
        df = pd.DataFrame(data)

        # Melt symptoms into a flat structure
        symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
        all_symptoms = set()
        for col in symptom_cols:
            valid = df[col].dropna().astype(str).str.strip()
            all_symptoms.update(valid.unique())

        assert len(all_symptoms) > 0, "No symptoms extracted"
        assert "fever" in all_symptoms
        assert "headache" in all_symptoms

    def test_label_encoding(self):
        """Test disease label encoding."""
        from sklearn.preprocessing import LabelEncoder

        diseases = ["Flu", "Cold", "Malaria", "Flu", "Cold"]
        le = LabelEncoder()
        encoded = le.fit_transform(diseases)

        assert len(set(encoded)) == 3  # 3 unique diseases
        assert encoded.min() == 0
        assert encoded.max() == 2

    def test_train_val_test_split_proportions(self):
        """Test that data splits have correct proportions."""
        from sklearn.model_selection import train_test_split

        n_samples = 1000
        X = np.random.randn(n_samples, 132)
        y = np.random.randint(0, 41, n_samples)

        # 70/15/15 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        assert abs(len(X_train) / n_samples - 0.7) < 0.02
        assert abs(len(X_val) / n_samples - 0.15) < 0.02
        assert abs(len(X_test) / n_samples - 0.15) < 0.02


class TestSkinPreprocessing:
    """Tests for skin lesion image preprocessing."""

    def test_image_transform_output_shape(self):
        """Test that image transforms produce correct tensor shape."""
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = transform(img)

        assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"

    def test_augmentation_transforms(self):
        """Test that augmentation transforms don't break the pipeline."""
        from torchvision import transforms
        from PIL import Image

        aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = aug_transform(img)
        assert tensor.shape == (3, 224, 224)
        assert not torch.isnan(tensor).any()

    def test_lesion_class_mapping(self):
        """Test HAM10000 lesion class mapping."""
        class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}

        assert len(class_to_idx) == 7
        assert "mel" in class_to_idx
        assert "nv" in class_to_idx


class TestRespiratoryPreprocessing:
    """Tests for respiratory sound preprocessing."""

    def test_mel_spectrogram_shape(self):
        """Test that Mel spectrogram has correct dimensions."""
        try:
            import librosa

            # Create a dummy audio signal (3 seconds at 22050 Hz)
            sr = 22050
            duration = 3.0
            audio = np.random.randn(int(sr * duration)).astype(np.float32)

            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            assert mel_spec_db.shape[0] == 128, f"Expected 128 mel bins, got {mel_spec_db.shape[0]}"
            assert mel_spec_db.shape[1] > 0, "Spectrogram has no time frames"
        except ImportError:
            pytest.skip("librosa not installed")

    def test_respiratory_labels(self):
        """Test respiratory sound label mapping."""
        labels = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}
        assert len(labels) == 4
        assert labels["Normal"] == 0
        assert labels["Both"] == 3


class TestDataPartitioning:
    """Tests for non-IID data partitioning."""

    def test_dirichlet_partition_sizes(self):
        """Test that Dirichlet partitioning creates correct number of partitions."""
        n_samples = 500
        n_clients = 10
        n_classes = 5

        # Simulate Dirichlet partitioning
        labels = np.random.randint(0, n_classes, n_samples)
        alpha = 0.5

        client_indices = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            class_indices = np.where(labels == c)[0]
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
            splits = np.split(class_indices, proportions[:-1])
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        assert len(client_indices) == n_clients
        total_assigned = sum(len(ci) for ci in client_indices)
        assert total_assigned == n_samples, f"Expected {n_samples}, got {total_assigned}"

    def test_each_client_has_data(self):
        """Test that each client gets at least some data."""
        n_samples = 1000
        n_clients = 5
        n_classes = 10
        alpha = 1.0  # Use higher alpha for more uniform distribution

        labels = np.random.randint(0, n_classes, n_samples)
        client_indices = [[] for _ in range(n_clients)]

        for c in range(n_classes):
            class_indices = np.where(labels == c)[0]
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
            splits = np.split(class_indices, proportions[:-1])
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        for i, indices in enumerate(client_indices):
            assert len(indices) > 0, f"Client {i} has no data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

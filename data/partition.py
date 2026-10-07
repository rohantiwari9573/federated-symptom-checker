"""
Data partitioning utilities for Federated Learning.
Implements non-IID Dirichlet distribution partitioning across N clients.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import argparse
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def partition_data_dirichlet(dataset: Dataset, num_clients: int, alpha: float, num_classes: int) -> List[Subset]:
    """
    Partition dataset into `num_clients` subsets using Dirichlet distribution.
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'metadata') and 'dx' in dataset.metadata:
        classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        class_to_idx = {c: i for i, c in enumerate(classes)}
        labels = np.array([class_to_idx.get(c, 0) for c in dataset.metadata['dx']])
    elif isinstance(dataset, torch.utils.data.TensorDataset):
        labels = dataset.tensors[1].numpy()
    else:
        logger.info("Iterating dataset to extract labels for partitioning...")
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
        
    min_size = 0
    min_require_size = 10
    N = len(labels)
    client_idcs = [[] for _ in range(num_clients)]

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, client_idcs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            client_idcs[j] = idx_batch[j]
            
    subsets = [Subset(dataset, idcs) for idcs in client_idcs]
    
    for i, idcs in enumerate(client_idcs):
        client_labels = labels[idcs]
        unique, counts = np.unique(client_labels, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"Client {i} data size: {len(idcs)}, label distribution: {dist}")
        
    return subsets

def get_client_dataloaders(dataset: Dataset, num_clients: int, alpha: float, batch_size: int, num_classes: int) -> List[DataLoader]:
    subsets = partition_data_dirichlet(dataset, num_clients, alpha, num_classes)
    return [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]

def create_client_dataloaders(dataset_name: str, num_clients: int, batch_size: int = 32, alpha: float = 0.5) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create client train and validation DataLoaders for federated simulation.
    Returns list of (train_loader, val_loader) for each client.
    """
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    if dataset_name == 'tabular':
        tabular_dir = PROJECT_ROOT / "data" / "processed" / "tabular"
        if not (tabular_dir / "X_train.pt").exists():
            from data.preprocess_tabular import preprocess_tabular
            preprocess_tabular()
        X_train = torch.load(tabular_dir / "X_train.pt")
        y_train = torch.load(tabular_dir / "y_train.pt")
        X_val = torch.load(tabular_dir / "X_val.pt")
        y_val = torch.load(tabular_dir / "y_val.pt")
        
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        val_ds = torch.utils.data.TensorDataset(X_val, y_val)
        num_classes = len(torch.unique(y_train))
        
    elif dataset_name == 'skin':
        from data.preprocess_skin import SkinLesionDataset, get_skin_transforms
        import pandas as pd
        skin_dir = PROJECT_ROOT / "data" / "raw" / "ham10000"
        meta_file = skin_dir / "HAM10000_metadata.csv"
        if not meta_file.exists():
            from data.preprocess_skin import main as pre_skin
            # run fallback
            from subprocess import run
            run(["python", str(PROJECT_ROOT / "data" / "preprocess_skin.py")])
        df = pd.read_csv(meta_file)
        img_dir = skin_dir / "images" if (skin_dir / "images").exists() else skin_dir
        train_ds = SkinLesionDataset(df, img_dir, transform=get_skin_transforms(train=True))
        val_ds = SkinLesionDataset(df, img_dir, transform=get_skin_transforms(train=False))
        num_classes = 7
        
    elif dataset_name == 'respiratory':
        from data.preprocess_respiratory import RespiratoryDataset
        resp_dir = PROJECT_ROOT / "data" / "raw" / "icbhi"
        if not resp_dir.exists() or len(list(resp_dir.rglob("*.wav"))) == 0:
            from subprocess import run
            run(["python", str(PROJECT_ROOT / "data" / "preprocess_respiratory.py")])
        train_ds = RespiratoryDataset(resp_dir)
        val_ds = RespiratoryDataset(resp_dir)
        num_classes = 4
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
    train_loaders = get_client_dataloaders(train_ds, num_clients, alpha, batch_size, num_classes)
    val_loaders = get_client_dataloaders(val_ds, num_clients, alpha, batch_size, num_classes)
    
    return list(zip(train_loaders, val_loaders))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partitioning utility")
    parser.add_argument('--dataset', type=str, default='tabular', choices=['tabular', 'skin', 'respiratory'])
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    logger.info(f"Mock partitioning for {args.dataset} with {args.num_clients} clients, alpha={args.alpha}")

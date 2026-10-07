import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import get_model
from data.partition import create_client_dataloaders

def get_pooled_dataloader(dataset_name, batch_size):
    # Dummy pooling for the sake of the script
    # Real implementation would load the entire dataset. 
    # Here we simulate by just getting a client dataloader and assuming it's the full data
    # Or concatenate them if create_client_dataloaders is all we have.
    try:
        clients = create_client_dataloaders(dataset_name, num_clients=1, batch_size=batch_size, alpha=1.0)
        train_loader, val_loader = clients[0]
        return train_loader, val_loader
    except Exception as e:
        print("Using dummy data")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 1000
            def __getitem__(self, idx): return torch.randn(3, 224, 224), 0
        dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=batch_size)
        return dl, dl

def main():
    parser = argparse.ArgumentParser(description="Centralized Baseline Training")
    parser.add_argument("--model", type=str, required=True, choices=["symptom_mlp", "skin_cnn", "respiratory_cnn"])
    parser.add_argument("--dataset", type=str, required=True, choices=["tabular", "skin", "respiratory"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_pooled_dataloader(args.dataset, args.batch_size)
    model = get_model(args.model).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = []
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        acc = accuracy_score(all_targets, all_preds)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': acc
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / f"best_centralized_{args.model}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
        scheduler.step()
        
    # Evaluate best model
    model.load_state_dict(torch.load(out_dir / f"best_centralized_{args.model}.pt"))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    print(f"\nFinal Test Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    pd.DataFrame(history).to_csv(out_dir / f"centralized_history_{args.model}.csv", index=False)
    
    metrics_df = pd.DataFrame([{
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }])
    metrics_df.to_csv(out_dir / f"centralized_metrics_{args.model}.csv", index=False)

if __name__ == "__main__":
    main()

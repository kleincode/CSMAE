from typing import Callable, Dict, Optional, List, Tuple
import numpy as np
import pickle
import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from eval_utils import get_ben_report, read_data

def prepare_dataloader(X: np.ndarray, Y: np.ndarray, device: torch.device, **dataloader_kwargs) -> torch.utils.data.DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

def eval(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module) -> Tuple[float, np.ndarray]:
    model.eval()
    loss = 0.0
    preds = []
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            Y_pred = model(X_batch)
            loss += criterion(Y_pred, Y_batch).item()
            preds.append(Y_pred.cpu().detach().numpy())
    return loss / len(dataloader), np.concatenate(preds)

def create_model(n_features: int, n_classes: int, n_layers: int) -> nn.Module:
    if n_layers == 1:
        return nn.Linear(n_features, n_classes)
    elif n_layers == 4:
        return nn.Sequential(
            nn.Linear(n_features, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )
    else:
        raise ValueError(f"Unsupported number of layers: {n_layers}. Choose 1 or 4")

def mlp_classifier(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    out_folder: Path,
    epochs: int = 100,
    n_layers: int = 1,
    early_stopping: Optional[int] = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 64,
    optim_supplier: Callable[[nn.Module], torch.optim.Optimizer] = lambda model: torch.optim.Adam(model.parameters()),
    criterion: nn.Module = nn.CrossEntropyLoss(),
    classes: int = 19,
    verbose: bool = True,
) -> Dict[str, object]:
    print("Classes:", classes)
    # Read data
    train_keys, Y_train, X_train = read_data(train_file, classes)
    assert len(train_keys) == len(Y_train) == len(X_train), "train dimensions mismatch"
    val_keys, Y_val, X_val = read_data(val_file, classes)
    assert len(val_keys) == len(Y_val) == len(X_val), "val dimensions mismatch"
    test_keys, Y_test, X_test = read_data(test_file, classes)
    assert len(test_keys) == len(Y_test) == len(X_test), "test dimensions mismatch"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "feature dimensions mismatch"
    assert Y_train.shape[1] == Y_val.shape[1] == Y_test.shape[1], "label dimensions mismatch"
    n_features = X_train.shape[1]
    n_classes = Y_train.shape[1]
    if verbose:
        print(f"Loaded {len(train_keys)} training, {len(val_keys)} validation, and {len(test_keys)} test samples")
        print(f"Feature dimension: {n_features}, label dimension: {n_classes}, device: {device}, batch size: {batch_size}")
    
    # Prepare data loaders
    train_dataloader = prepare_dataloader(X_train, Y_train, device, batch_size=batch_size, shuffle=True)
    val_dataloader = prepare_dataloader(X_val, Y_val, device, batch_size=batch_size)
    test_dataloader = prepare_dataloader(X_test, Y_test, device, batch_size=batch_size)
    
    # Training
    model = create_model(n_features, n_classes, n_layers).to(device)
    if verbose:
        print("Model:", model)
    optimizer = optim_supplier(model)
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_epoch: Optional[int] = None
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")
        
        # Validation
        val_loss, _ = eval(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
        
        # Early stopping
        if best_epoch is None or val_loss < val_losses[best_epoch]:
            best_epoch = epoch
        elif early_stopping is not None and epoch - best_epoch >= early_stopping:
            print(f"Stopping early at epoch {epoch}")
            break
    
    # Testing
    test_loss, Y_test_scores = eval(model, test_dataloader, criterion)
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model
    out_folder.mkdir(parents=True, exist_ok=True)
    model_file = out_folder / "model.pt"
    torch.save(model.state_dict(), model_file)
    if verbose:
        print(f"Model saved to {model_file}")
    
    # Make plots
    x = np.arange(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(1, len(train_losses))
    plt.legend()
    plt.savefig(out_folder / "loss.svg")
    
    # Report
    print(f"Best epoch: {best_epoch}")
    report, main_metrics = get_ben_report(Y_test, Y_test_scores)
    if verbose:
        print(report)
        print(main_metrics)
    report.to_csv(out_folder / "report.csv")
    with open(out_folder / "summary.txt", "w") as f:
        f.write(main_metrics)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "report": report,
        "main_metrics": main_metrics
    }

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("train_file", type=Path)
    parser.add_argument("val_file", type=Path)
    parser.add_argument("test_file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--classes", type=int, default=19)
    args = parser.parse_args()
    mlp_classifier(
        args.train_file,
        args.val_file,
        args.test_file,
        args.out,
        epochs=args.epochs,
        n_layers=args.n_layers,
        early_stopping=args.early_stopping,
        device=args.device,
        batch_size=args.batch_size,
        classes=args.classes,
        verbose=not args.silent,
    )

if __name__ == "__main__":
    main()
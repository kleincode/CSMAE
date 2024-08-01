"""
This script trains a baseline model (ResNet50, etc.) on the Sentinel-1 BEN19 data.
"""
from typing import Literal, Tuple, List, Dict, Optional
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, resnet18
import torchvision.transforms as transforms
import torch.nn as nn
from csv import reader as CSVReader
from eval_utils import get_ben_report
from run_model import BEN19_LABELS
from src.bigearthnet_dataset.BEN_lmdb_s1 import BENLMDBS1Reader
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import fbeta_score, hamming_loss

from src.bigearthnet_dataset.BEN_lmdb_utils import band_combi_to_mean_std

def feature_list_to_onehot(lst: List[str], label_map: Dict[str, int] = BEN19_LABELS) -> torch.Tensor:
    # all the valid ones and one additional if none are valid
    res = torch.zeros(len(label_map))
    for label in lst:
        res[label_map[label]] = 1
    assert res.sum() >= 1, "Result Tensor is all zeros - this is not allowed"
    return res

class BENS1Dataset(Dataset):
    def __init__(self, csv_file: Path, ben_reader: BENLMDBS1Reader, transform_pipeline: transforms.Compose = transforms.Compose([])):
        self.ben_reader = ben_reader
        self.transform_pipeline = transform_pipeline
        with open(csv_file, "r") as f:
            csv_reader = CSVReader(f)
            self.keys = [s1 for s2, s1 in csv_reader]
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.ben_reader.read(self.keys[idx])
        timg = torch.stack([torch.tensor(patch.bandVH.data), torch.tensor(patch.bandVV.data)], dim=0)
        assert timg.shape == (2, 120, 120)
        timg = self.transform_pipeline(timg)
        tlabel = feature_list_to_onehot(patch.new_labels) # type: ignore
        assert tlabel.shape == (19,)
        return timg, tlabel


def eval(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for X_batch, Y_batch in tqdm(dataloader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            Y_pred = model(X_batch)
            loss += criterion(Y_pred, Y_batch).item()
            preds.append(Y_pred.cpu().detach().numpy())
            labels.append(Y_batch.cpu().detach().numpy())
    return loss / len(dataloader), np.concatenate(preds), np.concatenate(labels)

def train_baseline(
    splits_dir: Path = Path("BigEarthNet-MM_19-classes_models/splits"),
    batch_size: int = 64,
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    epochs: int = 10,
    out_folder: Path = Path("trained_models/resnet50_10e"),
    save_every_epochs: int = 2,
    architecture: Literal["resnet50", "resnet18"] = "resnet50"
):
    torch.manual_seed(42)
    np.random.seed(42)
    reader = BENLMDBS1Reader(
        lmdb_dir="csmae_data/BigEarthNetEncoded.lmdb",
        label_type="new",
        image_size=(2, 120, 120),
        bands=2,
    )
    out_folder.mkdir(parents=True, exist_ok=True)
    
    mean, std = band_combi_to_mean_std(2)
    train_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_pipeline = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = BENS1Dataset(splits_dir / "train.csv", reader, train_pipeline)
    val_dataset = BENS1Dataset(splits_dir / "val.csv", reader, test_pipeline)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    if architecture == "resnet50":
        model = resnet50(pretrained=False, num_classes=19)
    elif architecture == "resnet18":
        model = resnet18(pretrained=False, num_classes=19)
    
    # we need to set the input channels to 2, refer to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L197C48-L197C97
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.to(device)
    
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses, val_losses, val_f2s, val_hls = [], [], [], []
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        pb = tqdm(train_loader)
        for X_batch, Y_batch in pb:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optim.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optim.step()
            loss_item = loss.item()
            train_loss += loss_item
            pb.set_description(f"loss={loss_item:.5f}")
        train_loss /= len(train_loader)
        val_loss, val_preds, val_labels = eval(model, val_loader, criterion, device)
        val_thresholded = val_preds > 0
        val_f2 = fbeta_score(val_labels, val_thresholded, average="samples", beta=2)
        val_hl = hamming_loss(val_labels, val_thresholded)
        
        print(f"Epoch {epoch}: train_loss = {train_loss:.5f} val_loss = {val_loss:.5f} val_f2 (sample avg) = {val_f2} val_hl = {val_hl}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f2s.append(val_f2)
        val_hls.append(val_hl)
        
        if epoch % save_every_epochs == 0 or epoch == epochs:
            model_file = out_folder / f"epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_file)
            print(f"Model saved to {model_file}")
            # Save losses + val f2
            with open(out_folder / "losses.csv", "w") as f:
                f.write("epoch,train_loss,val_loss,val_f2,val_hl\n")
                for i in range(len(train_losses)):
                    f.write(f"{i+1},{train_losses[i]},{val_losses[i]},{val_f2s[i]},{val_hls[i]}\n")
    
    #model.load_state_dict(torch.load(out_folder / f"epoch_20.pth"))
    ## TESTING
    test_dataset = BENS1Dataset(splits_dir / "test.csv", reader, test_pipeline)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, Y_test_scores, Y_test = eval(model, test_loader, criterion, device)
    assert Y_test_scores is not None and Y_test is not None
    print(f"Test loss: {test_loss:.4f}")
    
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
    report, main_metrics = get_ben_report(Y_test, Y_test_scores)
    print(report)
    print(main_metrics)
    report.to_csv(out_folder / "report.csv")
    with open(out_folder / "summary.txt", "w") as f:
        f.write(main_metrics)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_loss,
        "report": report,
        "main_metrics": main_metrics
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", type=Path, default=Path("BigEarthNet-MM_19-classes_models/splits"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_folder", type=Path, default=Path("trained_models/resnet18"))
    parser.add_argument("--save_every_epochs", type=int, default=2)
    parser.add_argument("--architecture", type=str, default="resnet50")
    args = parser.parse_args()
    train_baseline(
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        device=args.device,
        epochs=args.epochs,
        out_folder=args.out_folder,
        save_every_epochs=args.save_every_epochs,
        architecture=args.architecture
    )
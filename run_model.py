from typing import List, Dict
from argparse import ArgumentParser
from time import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from csv import reader as CSVReader
from src.csmae_backbone import CSMAEBackbone
from src.bigearthnet_dataset.BEN_lmdb_s1 import BENLMDBS1Reader
from omegaconf import OmegaConf
from typing import Dict, Callable, Tuple
from  pathlib import Path
import torch
import glob
import os

from src.augmentations import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
)
from src.bigearthnet_dataset.BEN_lmdb_utils import band_combi_to_mean_std

from src.utils import Messages
from src.vit_cmmae import CrossModalMaskedAutoencoderViT, vit_tiny, vit_small, vit_base, vit_large

TransformFunction = Callable[[np.ndarray], torch.Tensor]

BEN43_LABELS = {
    "Continuous urban fabric": 0,
    "Discontinuous urban fabric": 1,
    "Industrial or commercial units": 2,
    "Road and rail networks and associated land": 3,
    "Port areas": 4,
    "Airports": 5,
    "Mineral extraction sites": 6,
    "Dump sites": 7,
    "Construction sites": 8,
    "Green urban areas": 9,
    "Sport and leisure facilities": 10,
    "Non-irrigated arable land": 11,
    "Permanently irrigated land": 12,
    "Rice fields": 13,
    "Vineyards": 14,
    "Fruit trees and berry plantations": 15,
    "Olive groves": 16,
    "Pastures": 17,
    "Annual crops associated with permanent crops": 18,
    "Complex cultivation patterns": 19,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
    "Agro-forestry areas": 21,
    "Broad-leaved forest": 22,
    "Coniferous forest": 23,
    "Mixed forest": 24,
    "Natural grassland": 25,
    "Moors and heathland": 26,
    "Sclerophyllous vegetation": 27,
    "Transitional woodland/shrub": 28,
    "Beaches, dunes, sands": 29,
    "Bare rock": 30,
    "Sparsely vegetated areas": 31,
    "Burnt areas": 32,
    "Inland marshes": 33,
    "Peatbogs": 34,
    "Salt marshes": 35,
    "Salines": 36,
    "Intertidal flats": 37,
    "Water courses": 38,
    "Water bodies": 39,
    "Coastal lagoons": 40,
    "Estuaries": 41,
    "Sea and ocean": 42
}

BEN19_LABELS = {
    "Urban fabric": 0,
    "Industrial or commercial units": 1,
    "Arable land": 2,
    "Permanent crops": 3,
    "Pastures": 4,
    "Complex cultivation patterns": 5,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
    "Agro-forestry areas": 7,
    "Broad-leaved forest": 8,
    "Coniferous forest": 9,
    "Mixed forest": 10,
    "Natural grassland and sparsely vegetated areas": 11,
    "Moors, heathland and sclerophyllous vegetation": 12,
    "Transitional woodland, shrub": 13,
    "Beaches, dunes, sands": 14,
    "Inland wetlands": 15,
    "Coastal wetlands": 16,
    "Inland waters": 17,
    "Marine waters": 18
}

def feature_list_to_onehot(lst: List[str], label_map: Dict[str, int] = BEN19_LABELS) -> np.ndarray:
    # all the valid ones and one additional if none are valid
    res = np.zeros(len(label_map))
    for label in lst:
        res[label_map[label]] = 1
    assert res.sum() >= 1, "Result Tensor is all zeros - this is not allowed"
    return res

def load_model_legacy(model_id: str, device: str) -> Tuple[CrossModalMaskedAutoencoderViT, TransformFunction]:
    """Loads a trained model from a checkpoint, returning the model and the data augmentation pipeline.
    The model should be in ./trained_models/{model_id} with the following files:
    - *.ckpt: exactly one checkpoint file
    - cfg.yaml: configuration file

    Args:
        model_id (str): 8-character-id of model name to be evaluated. See under ./trained_models
        device (int, optional): GPU device number. Defaults to 0.
    
    Returns:
        Tuple[CrossModalMaskedAutoencoderViT, TransformFunction]: Tuple containing the model and the data augmentation function.
    """
    # Copied and modified from retrieval.py
    path_to_model = f"./trained_models/{model_id}"
    cfg = OmegaConf.load(f"{path_to_model}/cfg.yaml")
    OmegaConf.set_struct(cfg, False)
    ckpts = glob.glob(f'{path_to_model}/*.ckpt')

    ckpt_path = ckpts[0]
    assert len(ckpts) == 1
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    model_config_path = os.path.dirname(ckpt_path) + "/cfg.yaml"
    model_cfg = OmegaConf.load(model_config_path)

    # overwrite backbone config to match the one used for pre-training
    cfg.backbone = model_cfg.backbone
    
    # initialize backbone
    size_2_vit: Dict[str, Callable[[], CrossModalMaskedAutoencoderViT]] = {
        'vit_tiny': vit_tiny,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
    }
    backbone = size_2_vit[cfg.backbone.name](**cfg.backbone.kwargs)

    # load ckpt weights into backbone
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    ret = backbone.load_state_dict(state, strict=True)
    Messages.hint(f"Loaded checkpoint ({ret})")

    # Build data augmentation pipeline, disabling all non-normalizing transforms
    pipelines = []
    for aug_cfg in cfg.augmentations:
        aug_cfg.rrc.enabled = False
        aug_cfg.horizontal_flip.prob = .0
        aug_cfg.vertical_flip.prob = .0
        pipelines.append(
            NCropAugmentation(
                build_transform_pipeline(cfg.data.dataset, aug_cfg, cfg), aug_cfg.num_crops
            )
        )
    transform = FullTransformPipeline(pipelines) # type: ignore

    backbone.eval()
    backbone.to(device)
    
    def full_transform(img: np.ndarray):
        # From BEN_DataModule_LMDB_Encoder.py, line 99ff
        img = np.transpose(img,(2,1,0)) # (C, W, H) -> (H, W, C)
        assert img.shape == (120, 120, 12), f"Expected shape (120, 120, 12), got {img.shape}"
        assert transform is not None
        img = transform(img) # type: ignore
        assert len(img) == 1
        assert img[0].shape == (12, 120, 120), f"Expected shape (12, 120, 120), got {img[0].shape}"
        return img[0]
    
    return backbone, full_transform

def load_model_new(csmae_variant: str, device: str) -> Tuple[CSMAEBackbone, TransformFunction]:
    """Loads a trained model from a checkpoint and returns the data normalization function.
    The model should be in ./checkpoints/{csmae_variant} with the following files:
    - weights.ckpt: exactly one checkpoint file
    - cfg.yaml: configuration file

    Args:
        csmae_variant (str): 4-characted lowercase CSMAE variant (cecd, cesd, secd, sesd)
        device (int, optional): GPU device number. Defaults to 0.
    
    Returns:
        Tuple[CSMAEBackbone, TransformFunction]: Loaded model and function that normalizes data.
    """
    # According to new README.md
    cfg = OmegaConf.load(f'./checkpoints/{csmae_variant}/cfg.yaml')
    model = CSMAEBackbone(**cfg.kwargs)

    state_dict = torch.load(f'./checkpoints/{csmae_variant}/weights.ckpt', map_location="cpu")['state_dict']
    for k in list(state_dict.keys()):
        if "backbone" in k:
            state_dict[k.replace("backbone.", "")] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict, strict=True)
    
    mean, std = band_combi_to_mean_std(12)
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    print(f"{mean=} {std=}")

    def transform(img: np.ndarray) -> torch.Tensor:
        assert img.shape == (12, 120, 120), f"Expected shape (12, 120, 120), got {img.shape}"
        # From BEN_DataModule_LMDB_Encoder.py, line 99ff
        img = np.transpose(img,(2,1,0)) # (C, W, H) -> (H, W, C)
        assert img.shape == (120, 120, 12), f"Expected shape (120, 120, 12), got {img.shape}"
        return norm(img) # type: ignore

    return model.eval().to(device), transform

def main(args):
    if args.legacy:
        model, transform = load_model_legacy(args.model, args.device)
    else:
        model, transform = load_model_new(args.model, args.device)
    reader = BENLMDBS1Reader(
        lmdb_dir="csmae_data/BigEarthNetEncoded.lmdb",
        label_type="new",
        image_size=(2, 120, 120),
        bands=2,
    )
    batch_size = args.batch_size or 8

    input_tensor = torch.zeros((batch_size, 12, 120, 120), device=args.device)
    keys, labels19, labels43, outputs = [], [], [], []
    times = 0
    batch = 0
    if args.csv is None:
        iterator = reader.iterate()
    else:
        with open(args.csv, "r") as f:
            csv_reader = CSVReader(f)
            iterator = [(s1, None) for s2, s1 in csv_reader]
    for key, patch in tqdm(iterator):
        if patch is None:
            patch = reader.read(key)
        keys.append(key)
        labels19.append(feature_list_to_onehot(patch.new_labels, BEN19_LABELS)) # type: ignore
        labels43.append(feature_list_to_onehot(patch.labels, BEN43_LABELS)) # type: ignore
        img = np.zeros((12, 120, 120))
        img[10] = patch.bandVH.data
        img[11] = patch.bandVV.data
        input_tensor[batch] = transform(img)
        batch += 1
        if batch == batch_size:
            batch = 0
            start = time()
            output = model(input_tensor)
            times += time() - start
            out = output["s1"].cpu().detach().numpy()
            for b in range(batch_size):
                outputs.append(out[b])
    if batch > 0:
        start = time()
        output = model(input_tensor)
        times += time() - start
        out = output["s1"].cpu().detach().numpy()
        for b in range(batch):
            outputs.append(out[b])

    print(f"Average time per patch: {times / len(outputs)}s")
    print("Saving...")
    outputs = np.array(outputs)
    labels19 = np.array(labels19)
    labels43 = np.array(labels43)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump((keys, labels19, labels43, outputs), f)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="run_model",
        description="Run the model on the BigEarthNet dataset in LMDB representation",
        epilog="Hot Topics in CV Seminar, 2024"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="ID of the model to evaluate, e.g. CSMAE-CECD",
        required=True
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the CSV file containing the keys of the patches to evaluate (e.g. train split). If not provided, all patches are evaluated.",
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output file",
        default="output.pkl"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the legacy model loading approach (before update of README.md)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on",
        default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size to use",
        default=8
    )
    args = parser.parse_args()
    print("Using patches:", args.csv or "ALL")
    print("Output file:", args.output)
    print("Labels:", "BEN19 and BEN43")
    print("Device:", args.device)
    print("Batch size:", args.batch_size)
    print("Legacy model loader:", args.legacy)
    with torch.no_grad():
        main(args)
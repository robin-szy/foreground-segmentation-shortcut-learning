from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
import torchvision.transforms.functional as tv_functional


"""
PyTorch's Dataset and DataLoader:
Dataset stores the samples and their corresponding labels, and DataLoader wraps an 
iterable around the Dataset to enable easy access to the samples.

I prefer to have full control here, as each index is assigned one of the folders
By automatically creating the indices, I'm scared that the folders might be
differently arranged
"""
CLASSES = [
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Fungi",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Plantae",
    "Reptilia",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def parse_args():

    parser = argparse.ArgumentParser(description="10-class iNaturalist")

    # Experiment details
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs_10class")
    parser.add_argument("--seed", type=int, default=42)

    # Training or testing
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--final-train", action="store_true", default=True)

    # Debugging -> Subset
    parser.add_argument("--max-train-samples", type=int, default=5)#default=None)

    # Domains
    parser.add_argument("--train-domain", choices=["normal", "cropped"], default="normal")
    parser.add_argument("--eval-domains", nargs="+", choices=["normal", "cropped"], default=["normal"])

    parser.add_argument("--normal-split-dir", type=str, default="inaturalist_12K/splits/split_seed_42")
    parser.add_argument("--normal-root", type=str, default=".")
    parser.add_argument("--normal-strip-prefix", type=str, default=None)

    parser.add_argument("--cropped-split-dir", type=str, default="inaturalist_12K/splits/split_seed_42")
    parser.add_argument("--cropped-root", type=str, default="inaturalist_12K_cropped")
    parser.add_argument("--cropped-strip-prefix", type=str, default="inaturalist_12K/raw")

    # Model
    parser.add_argument("--model-type", choices=["custom", "resnet18", "resnet50", "efficientnet_b0"], default="custom")
    #parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for torchvision models")
    #parser.add_argument("--freeze-backbone", action="store_true", help="Only train classifier head for transfer models")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # Data
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--aug", choices=["random_resized_crop", "square_pad"], default="random_resized_crop")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")

    args = parser.parse_args()
    args.pin_memory = not args.no_pin_memory
    return args



class SquarePad:
    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        pad_right = max_side - w - pad_left
        pad_bottom = max_side - h - pad_top
        return tv_functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)



class CsvImageDataset(Dataset):
    """
    Reads split CSV rows with columns: filepath,label,split,source.

    - normal data usually works with root='.' and the CSV filepath unchanged.
    - cropped data often mirrors filenames but has another root. Use
      --cropped-strip-prefix to remove a leading prefix from CSV filepath.

    More details, see https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    It needs to have this structure, because DataLoader only accepts two different types of
    datasets: map-style and iterable-style. Below is a map-style one (__len__ and __getitem__
    necessary). More: https://docs.pytorch.org/docs/stable/data.html
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        transform=None,
        strip_prefix: Optional[str] = None,
    ):
        self.csv_file = Path(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.strip_prefix = strip_prefix.strip("/") if strip_prefix else None
        self.df = pd.read_csv(self.csv_file)

        required = {"filepath", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.csv_file} missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> Path:
        rel = rel_path.replace("\\", "/").strip("/")
        if self.strip_prefix and rel.startswith(self.strip_prefix + "/"):
            rel = rel[len(self.strip_prefix) + 1 :]
        return self.root_dir / rel

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(str(row["filepath"]))
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"CSV filepath: {row['filepath']}\n"
                f"root_dir: {self.root_dir}\n"
                f"strip_prefix: {self.strip_prefix}"
            )

        label_name = str(row["label"])
        if label_name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class label {label_name!r} in {self.csv_file}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, CLASS_TO_IDX[label_name]
        # DON'T DO THIS: return image.to("cuda"), label -> If this is together with num_workers -> Bad idea
        # "it is probably not a good idea to call .cuda() in the Dataset object, as it will have to move each
        # sample (rather than the batch) to GPU separately, incurring a lot of overhead."
        # https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# Overview transformations:
# https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
def get_transforms(img_size: int, aug: str, model_type: str):
    """
    Use ImageNet normalization for transfer learning.
    Simple normalization for custom CNN.
    See also:
        https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    """
    if model_type in {"resnet18", "resnet50", "efficientnet_b0"}:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    if aug == "random_resized_crop":
        print("Using random resized crop")
        train_tf = transforms.Compose([
            # The scale and ratio of the RandomResizedCrop is chosen with a uniform distribution (see code)
            # https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5), # Todo: Remove?
            transforms.ToTensor(),
            normalize,
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == "square_pad":
        print("Using resize and zero padding")
        train_tf = transforms.Compose([
            SquarePad(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        eval_tf = transforms.Compose([
            SquarePad(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown aug: {aug}")

    return train_tf, eval_tf


class CustomCNN(nn.Module):
    """Small self-made CNN. No transfer learning."""

    # Todo: Add more layers
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),   # Todo: Remove?
            # The adaptive avg pooling uses the downsampled image (e.g. at 14x14
            # pixels) and takes the mean. Then, the output is only 1 vector per
            # image. This vector contains a latent feature representation (here 256 features)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    # _block does not need "self", so we decorate it with @staticmethod
    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model_type: str, num_classes: int, dropout: float) -> nn.Module:
    if model_type == "custom":
        return CustomCNN(num_classes=num_classes, dropout=dropout)

    if model_type == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_type == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_type == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def freeze_model_parameters(model: nn.Module, model_type: str) -> None:
    """
    For transfer learning. If we use a pretrained model, we freeze the parameters,
    so they are not trained again. Then, we only learn the parameters of the last
    layer (which we added).
    """
    if model_type == "custom":
        return
    for p in model.parameters():
        p.requires_grad = False
    if model_type.startswith("resnet"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_type == "efficientnet_b0":
        for p in model.classifier.parameters():
            p.requires_grad = True


def make_domain_config(args, domain: str) -> Dict[str, Path | str | None]:
    if domain == "normal":
        return {
            "split_dir": Path(args.normal_split_dir),
            "root": Path(args.normal_root),
            "strip_prefix": args.normal_strip_prefix,
        }
    if domain == "cropped":
        return {
            "split_dir": Path(args.cropped_split_dir),
            "root": Path(args.cropped_root),
            "strip_prefix": args.cropped_strip_prefix,
        }
    raise ValueError(f"Unknown domain: {domain}")


# With dataclass decorator, there's no need to write def __init__, etc.
@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_accuracy: float
    macro_precision: float


def compute_macro_accuracy(confusion: np.ndarray) -> float:
    per_class = []
    for i in range(confusion.shape[0]):
        # Rows are true values. We take sum of entire row.
        denom = confusion[i].sum()
        if denom > 0:
            per_class.append(confusion[i, i] / denom)
            # Main diagonal has all true positives. Divide by total number.
            # Per class accuracy (recall) -> 80% of real bird were also classified so.
            # Recall TP / (TP + FN)
    return float(np.mean(per_class)) if per_class else 0.0


def compute_macro_precision(confusion):
    # Precision = TP / (TP + FP)
    per_class = []
    for i in range(confusion.shape[0]):
        denom = confusion[:, i].sum()
        if denom > 0:
            per_class.append(confusion[i, i] / denom)
    return float(np.mean(per_class))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[Metrics, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        for t, p in zip(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()):
            confusion[t, p] += 1

    return Metrics(total_loss / total, correct / total, compute_macro_accuracy(confusion), compute_macro_precision(confusion)), confusion


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def metrics_dict(prefix: str, m: Metrics) -> Dict[str, float]:
    return {
        f"{prefix}_loss": m.loss,
        f"{prefix}_accuracy": m.accuracy,
        f"{prefix}_macro_accuracy": m.macro_accuracy,
        f"{prefix}_macro_precision": m.macro_precision,
    }


def make_balanced_debug_subset(dataset, samples_per_class, seed):
    # Just for debugging on PC, since I don't have a GPU and one epoch takes ages.
    rng = np.random.default_rng(seed)

    # Case concatenated dataset (final_train) plus debugging
    if dataset is None:
        return None
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        labels = []
        for subdataset in dataset.datasets:
            labels.extend(subdataset.df["label"].tolist())
        labels = np.array(labels)

    else:
        labels = dataset.df["label"].to_numpy()

    indices = []
    for cls in CLASSES:
        cls_idx = np.where(labels == cls)[0]
        chosen = rng.choice(
            cls_idx,
            size=min(samples_per_class, len(cls_idx)),
            replace=False,
        )
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices)


def append_results_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)




def train_model(args) -> None:
    set_seed(args.seed)

    run_name = args.run_name or f"{args.model_type}_train-{args.train_domain}_seed{args.seed}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_json(out_dir / "args.json", vars(args))

    # Get the transformations and datasets
    train_tf, eval_tf = get_transforms(args.img_size, args.aug, args.model_type)

    # Datasets
    train_cfg = make_domain_config(args, args.train_domain)

    train_dataset = CsvImageDataset(
        csv_file=Path(train_cfg["split_dir"]) / "train.csv",
        root_dir=Path(train_cfg["root"]),
        strip_prefix=train_cfg["strip_prefix"],
        transform=train_tf,
    )

    if args.final_train:
        print("FINAL TRAINING MODE: full dataset, no early stopping.")
        val_dataset_raw = CsvImageDataset(
            csv_file=Path(train_cfg["split_dir"]) / "val.csv",
            root_dir=Path(train_cfg["root"]),
            strip_prefix=train_cfg["strip_prefix"],
            transform=train_tf,
        )
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset_raw])
        val_dataset = None
    else:
        val_dataset = CsvImageDataset(
            csv_file=Path(train_cfg["split_dir"]) / "val.csv",
            root_dir=Path(train_cfg["root"]),
            strip_prefix=train_cfg["strip_prefix"],
            transform=eval_tf,
        )

    # For debugging because one epoch takes very long
    if args.max_train_samples is not None:
        train_dataset = make_balanced_debug_subset(
            dataset=train_dataset,
            samples_per_class=args.max_train_samples,
            seed=args.seed,
        )
        val_dataset = make_balanced_debug_subset(
            dataset=val_dataset,
            samples_per_class=args.max_train_samples,
            seed=args.seed,
        )

    # Troubleshooting
    # Memory overload? Maybe it's due to setting num_workers, see here:
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # Also pin_memory can lead to errors if not enough RAM,
    # see: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,   # Multi-process data loading (how many subprocesses)
        pin_memory=args.pin_memory,     # PyTorch usually copies data from CPU memory to GPU (slower). With pin_memory
                                        # data is allocated in pinned (page-locked) RAM, GPU can transfer data more
                                        # efficiently via Direct Memory Access. Only do if enough RAM.
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if not args.final_train:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )

    # Get the model
    # Extra function to switch between own model and transfer learning
    model = build_model(args.model_type, num_classes=len(CLASSES), dropout=args.dropout)
    freeze_model_parameters(model, args.model_type)   # Freeze model parameters if transfer learning
    model.to(device)

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(args.optimizer)

    # Scaler due to AMP
    # Some operations, like convolutions, are much faster in float16 than in float32.
    # torch.cuda.amp provides some methods for mixed precision (some 32 float, some 16).
    # AMP = Automatic mixed precision.
    # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    # Since we use AMP, we also need gradient scaler:
    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=args.amp and device.type == "cuda"
    )

    best_val_acc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    best_checkpoint = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # AMP-aware training loop
        if args.amp and device.type == "cuda":
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                # Autocast takes care of weight updates with half precision and
                # avoids explosive loss (overflows).
                # It automatically chooses cheaper datatypes where safe
                # Structure below based on:
                # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
                with torch.amp.autocast(
                        "cuda",
                        enabled=args.amp and device.type == "cuda"
                ):
                    logits = model(images)
                    #assert logits.dtype is torch.float16
                    loss = criterion(logits, labels)
                    #assert loss.dtype is torch.float32
                # Exit autocast before backward(). Backward passes under autocast are not recommended
                # Backpropagation, but increase if they are tiny so they survive.
                scaler.scale(loss).backward()
                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)  # Unscale before
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)  # Update model weights. Unscales gradients internally, no need to unscale before
                scaler.update() # To maintain a good scaling factor for next iteration

                bs = labels.size(0)
                total_loss += loss.item() * bs
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += bs
                for t, p in zip(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()):
                    confusion[t, p] += 1
                train_m = Metrics(
                    total_loss / total,
                    correct / total,
                    compute_macro_accuracy(confusion),
                    compute_macro_precision(confusion),
                )
        else:
            # CPU
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                # Multiply total loss by batch size, because cross-entropy takes mean for batch
                # Why not just divide by number of batches? Because last batch might be smaller.
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

                for t, p in zip(labels.detach().cpu().numpy(), preds.detach().cpu().numpy()):
                    confusion[t, p] += 1

            train_m = Metrics(
                total_loss / total,
                correct / total,
                compute_macro_accuracy(confusion),
                compute_macro_precision(confusion)
            )

        if not args.final_train:
            val_m, val_conf = evaluate(model, val_loader, criterion, device)
        else:
            val_m, val_conf = None, None
        elapsed = time.time() - start

        row = {"epoch": epoch, "seconds": elapsed}
        row.update(metrics_dict("train", train_m))
        if not args.final_train:
            row.update(metrics_dict("val", val_m))
        history.append(row)

        if not args.final_train:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_m.loss:.4f}, acc {train_m.accuracy:.4f}, "
                f"m-recall {train_m.macro_accuracy:.4f}, m-prec {train_m.macro_precision:.4f} | "
                f"val loss {val_m.loss:.4f}, acc {val_m.accuracy:.4f}, "
                f"m-recall {val_m.macro_accuracy:.4f}, m-prec {val_m.macro_precision:.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} FINAL | "
                f"train loss {train_m.loss:.4f}, acc {train_m.accuracy:.4f}, "
                f"m-recall {train_m.macro_accuracy:.4f}, m-prec {train_m.macro_precision:.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

        # Early stopping
        if args.final_train:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "classes": CLASSES,
                    "args": vars(args),
                },
                best_checkpoint,
            )
        else:
            improved = val_m.accuracy > best_val_acc + args.min_delta
            if improved:
                best_val_acc = val_m.accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "classes": CLASSES,
                        "args": vars(args),
                        "val_metrics": asdict(val_m),
                    },
                    best_checkpoint,
                )
                np.savetxt(out_dir / "best_val_confusion.csv", val_conf,
                           fmt="%d", delimiter=",")
            else:
                epochs_without_improvement += 1

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

        summary_row = {
            "run_name": run_name,
            "model_type": args.model_type,
            "train_domain": args.train_domain,
            "eval_domains": ",".join(args.eval_domains),
            "aug": args.aug,
            "img_size": args.img_size,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "final_train": args.final_train,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_epoch": best_epoch if not args.final_train else epoch,
            "best_val_acc": best_val_acc if not args.final_train else None,
        }

        append_results_csv(Path(args.out_dir) / "results.csv", summary_row)


def test_model(args):

    if args.checkpoint is None:
        raise ValueError("--checkpoint required in test mode")
    device = torch.device(args.device)

    # Build model
    model = build_model(
        args.model_type,
        num_classes=len(CLASSES),
        dropout=args.dropout,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Transforms
    _, eval_tf = get_transforms(
        args.img_size,
        args.aug,
        args.model_type,
    )

    # Build test loaders
    test_loaders = {}

    for domain in args.eval_domains:

        cfg = make_domain_config(args, domain)
        test_dataset = CsvImageDataset(
            csv_file=Path(cfg["split_dir"]) / "test.csv",
            root_dir=Path(cfg["root"]),
            strip_prefix=cfg["strip_prefix"],
            transform=eval_tf,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        test_loaders[domain] = test_loader

    # Evaluate
    final_results = {}
    for domain, loader in test_loaders.items():

        metrics, confusion = evaluate(
            model,
            loader,
            criterion,
            device,
        )

        final_results[domain] = asdict(metrics)

        print(
            f"TEST {domain} | "
            f"loss {metrics.loss:.4f} | "
            f"acc {metrics.accuracy:.4f} | "
            f"m-recall {metrics.macro_accuracy:.4f} "
            f"m-prec {metrics.macro_precision:.4f}"
        )

        # Save predictions
        rows = []
        model.eval()

        with torch.no_grad():
            for images, labels in loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                for true_idx, pred_idx, prob_vec in zip(
                        labels.cpu().numpy(),
                        preds.cpu().numpy(),
                        probs.cpu().numpy(),
                ):
                    rows.append({
                        "true_label": CLASSES[int(true_idx)],
                        "pred_label": CLASSES[int(pred_idx)],
                        "correct": bool(true_idx == pred_idx),
                        "prob_pred": float(prob_vec[int(pred_idx)]),
                        "prob_true": float(prob_vec[int(true_idx)]),
                    })

        pred_df = pd.DataFrame(rows)
        pred_path = Path(args.out_dir) / f"test_predictions_{domain}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

        save_json(Path(args.out_dir) / "test_results.json", final_results)



if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)

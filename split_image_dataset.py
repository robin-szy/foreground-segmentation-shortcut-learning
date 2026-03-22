"""
I've created this script by using ChatGPT (model GPT 5.3). It does not only
split the training set to a test set, but also provides "views" for my laptop.
In Aurelien Geron's book on Machine Learning, I've read that I should not even
look into test dataset. So, I've created the views.
"""


import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# ----------------------------
# Configuration
# ----------------------------
SEED = 42
N_TEST_PER_CLASS = 200

PROJECT_ROOT = Path(".").resolve()
RAW_TRAIN_DIR = PROJECT_ROOT / "inaturalist_12K" / "raw" / "train"
RAW_VAL_DIR = PROJECT_ROOT / "inaturalist_12K" / "raw" / "val"

SPLIT_NAME = f"split_seed_{SEED}"
SPLIT_DIR = PROJECT_ROOT / "inaturalist_12K" / "splits" / SPLIT_NAME
VIEW_DIR = PROJECT_ROOT / "inaturalist_12K" / "views" / SPLIT_NAME

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CREATE_VIEWS = True
CREATE_TEST_VIEW = True   # keep False to avoid peeking


def list_images(class_dir: Path):
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def to_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def collect_split_rows():
    rng = random.Random(SEED)

    if not RAW_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {RAW_TRAIN_DIR}")
    if not RAW_VAL_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {RAW_VAL_DIR}")

    train_classes = sorted([p.name for p in RAW_TRAIN_DIR.iterdir() if p.is_dir()])
    val_classes = sorted([p.name for p in RAW_VAL_DIR.iterdir() if p.is_dir()])

    if train_classes != val_classes:
        raise ValueError(
            f"Class mismatch between train and val.\n"
            f"train: {train_classes}\nval:   {val_classes}"
        )

    rows = {"train": [], "val": [], "test": []}
    summary = defaultdict(dict)

    for cls in train_classes:
        train_class_dir = RAW_TRAIN_DIR / cls
        val_class_dir = RAW_VAL_DIR / cls

        train_imgs = list_images(train_class_dir)
        val_imgs = list_images(val_class_dir)

        if len(train_imgs) != 1000:
            print(f"Warning: class {cls} has {len(train_imgs)} train images, expected 1000")
        if len(val_imgs) != 200:
            print(f"Warning: class {cls} has {len(val_imgs)} val images, expected 200")

        rng.shuffle(train_imgs)

        test_imgs = train_imgs[:N_TEST_PER_CLASS]
        new_train_imgs = train_imgs[N_TEST_PER_CLASS:]

        for img in new_train_imgs:
            rows["train"].append({
                "filepath": to_relative(img, PROJECT_ROOT),
                "label": cls,
                "split": "train",
                "source": "train"
            })

        for img in test_imgs:
            rows["test"].append({
                "filepath": to_relative(img, PROJECT_ROOT),
                "label": cls,
                "split": "test",
                "source": "train"
            })

        for img in val_imgs:
            rows["val"].append({
                "filepath": to_relative(img, PROJECT_ROOT),
                "label": cls,
                "split": "val",
                "source": "val"
            })

        summary[cls]["train"] = len(new_train_imgs)
        summary[cls]["test"] = len(test_imgs)
        summary[cls]["val"] = len(val_imgs)

    return rows, summary


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)


def create_symlink_view(rows, split_name: str):
    if split_name == "test" and not CREATE_TEST_VIEW:
        return

    for row in rows:
        src = PROJECT_ROOT / row["filepath"]
        dst = VIEW_DIR / split_name / row["label"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Fallback to copying if symlinks are unavailable on the system
            import shutil
            shutil.copy2(src, dst)


def sanity_check(rows):
    seen = set()
    for split_name, split_rows in rows.items():
        for row in split_rows:
            key = row["filepath"]
            if key in seen:
                raise ValueError(f"Leakage detected: {key} appears in multiple splits")
            seen.add(key)

    print("Sanity check passed: no file appears in more than one split.")


def main():
    rows, summary = collect_split_rows()
    sanity_check(rows)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(rows["train"], SPLIT_DIR / "train.csv")
    write_csv(rows["val"], SPLIT_DIR / "val.csv")
    write_csv(rows["test"], SPLIT_DIR / "test.csv")

    with (SPLIT_DIR / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if CREATE_VIEWS:
        create_symlink_view(rows["train"], "train")
        create_symlink_view(rows["val"], "val")
        create_symlink_view(rows["test"], "test")

    print(f"Created split in: {SPLIT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
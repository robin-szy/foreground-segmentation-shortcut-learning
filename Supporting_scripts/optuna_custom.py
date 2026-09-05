import argparse
import optuna
from argparse import Namespace
from pathlib import Path
import pandas as pd

from train_10_class_classifier import train_model

def objective(trial):
    args = Namespace(
        run_name=f"optuna_trial_{trial.number}",
        out_dir="runs_optuna",
        seed=42,
        mode="train",
        checkpoint=None,
        final_train=False,
        max_train_samples=None,

        train_domain="normal",
        eval_domains=["normal"],
        normal_split_dir=ARGS.normal_split_dir,
        normal_root=ARGS.normal_root,
        normal_strip_prefix=None,
        cropped_split_dir=ARGS.normal_split_dir,
        cropped_root="inaturalist_12K_cropped",
        cropped_strip_prefix="inaturalist_12K/raw",

        model_type="custom",
        epochs=60,
        patience=10,
        min_delta=1e-4,

        batch_size=trial.suggest_categorical("batch_size", [32, 64]),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.2, 0.5),
        optimizer=trial.suggest_categorical("optimizer", ["adamw", "nadam"]),
        momentum=0.9,
        grad_clip=1.0,

        img_size=224,
        aug=trial.suggest_categorical("aug", ["random_resized_crop", "square_pad"]),
        num_workers=4,
        no_pin_memory=False,
        pin_memory=True,

        use_segmented=False,
        segmented_root=ARGS.segmented_root,
        segmented_prob=0.0,
        segmented_val=False,

        device="cuda",
        amp=True,
        resume=False,
    )

    train_model(args)

    results = pd.read_csv(Path(args.out_dir) / "results.csv")
    row = results[results["run_name"] == args.run_name].iloc[-1]
    return float(row["best_val_acc"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-root", required=True)
    parser.add_argument("--normal-split-dir", required=True)
    parser.add_argument("--segmented-root", default="inaturalist_12K_segmented/segm_full_train_val_vitb_pps16")
    parser.add_argument("--n-trials", type=int, default=8)
    global ARGS
    ARGS = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ARGS.n_trials)

    print("BEST VALUE:", study.best_value)
    print("BEST PARAMS:", study.best_params)
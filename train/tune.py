"""
tune_resume.py — same as tune.py but supports resuming partial runs.
Checks CSV for already-completed trials and only runs the remaining ones.

Once current runs finish, replace tune.py with this file:
    mv train/tune_resume.py train/tune.py
"""

import os
import json
import copy
import argparse

import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

from pytorch_modules.utils.utils import load_config
from train.train_RNA_model_HP_tune import run_trial
import fcntl
import csv

CSV_PATH = "./hp_results/tuning_history.csv"
FIELDNAMES = [
    "rna_embedding_key", "arch", "loss_name", "split_idx",
    "trial_number", "val_loss", "lr", "weight_decay", "penalty_weight", "dropout_rate"
]

def get_effective_key(rna_embedding_key, n_HVGs):
    """Fuse RNA_seq with nHVGs for CSV/json tracking."""
    if rna_embedding_key == "RNA_seq":
        return f"RNA_seq_{n_HVGs}"
    return rna_embedding_key

def get_trial_callback(args):
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    def trial_callback(study, trial):
        print(
            f"  Trial {trial.number:>3} | "
            f"val_loss: {trial.value:.4f} | "
            f"best: {study.best_value:.4f} | "
            f"params: {trial.params}"
        )
        row = {
            "rna_embedding_key": get_effective_key(args.rna_embedding_key, args.n_HVGs),
            "arch": args.arch,
            "loss_name": args.loss_name,
            "split_idx": args.split_idx,
            "trial_number": trial.number,
            "val_loss": trial.value,
            "lr": trial.params.get("lr", ""),
            "weight_decay": trial.params.get("weight_decay", ""),
            "penalty_weight": trial.params.get("penalty_weight", ""),
            "dropout_rate": trial.params.get("dropout_rate", ""),
        }
        with open(CSV_PATH, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)
            fcntl.flock(f, fcntl.LOCK_UN)

    return trial_callback


def make_objective(config, args, split_idx):
    def objective(trial):
        hp_overrides = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "penalty_weight": trial.suggest_float("penalty_weight", 1e-4, 1e-2, log=True),
        }
        if args.arch != "Linear":
            hp_overrides["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)

        trial_args = copy.copy(args)
        trial_args.split_idx = split_idx
        trial_config = load_config(args.config_path)
        return run_trial(trial_config, trial_args, hp_overrides)

    return objective

def get_completed_trials(args):
    """How many trials already done for this config in the CSV."""
    if not os.path.exists(CSV_PATH):
        return 0
    try:
        df = pd.read_csv(CSV_PATH)
        return df[
            (df["rna_embedding_key"] == get_effective_key(args.rna_embedding_key, args.n_HVGs)) &
            (df["arch"] == args.arch) &
            (df["loss_name"] == args.loss_name) &
            (df["split_idx"] == args.split_idx)
        ].shape[0]
    except Exception:
        return 0


def main(args):
    completed = get_completed_trials(args)
    remaining = args.n_trials - completed

    if remaining <= 0:
        print(f"SKIP — already complete ({completed} trials): "
              f"{get_effective_key(args.rna_embedding_key, args.n_HVGs)} | {args.arch} | {args.loss_name} | split {args.split_idx}")
        return

    print(f"\n=== Tuning {get_effective_key(args.rna_embedding_key, args.n_HVGs)} | {args.arch} | {args.loss_name} | split {args.split_idx} "
          f"({completed} done, {remaining} remaining) ===")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(load_config(args.config_path), args, args.split_idx),
        n_trials=remaining,  # only run remaining trials
        callbacks=[get_trial_callback(args)],
        show_progress_bar=False,
    )

    print(f"  Best val loss: {study.best_value:.4f}")
    print(f"  Best params:   {study.best_params}")

    # Only save json if now fully complete
    total_completed = get_completed_trials(args)
    if total_completed >= args.n_trials:
        # Get best params from full CSV history for this config
        df = pd.read_csv(CSV_PATH)
        mask = (
            (df["rna_embedding_key"] == get_effective_key(args.rna_embedding_key, args.n_HVGs)) &
            (df["arch"] == args.arch) &
            (df["loss_name"] == args.loss_name) &
            (df["split_idx"] == args.split_idx)
        )
        best_row = df[mask].loc[df[mask]["val_loss"].idxmin()]
        best_params = {
            "lr": best_row["lr"],
            "weight_decay": best_row["weight_decay"],
            "penalty_weight": best_row["penalty_weight"],
        }
        if args.arch != "Linear":
            best_params["dropout_rate"] = best_row["dropout_rate"]

        os.makedirs("./hp_results", exist_ok=True)
        out_path = f"./hp_results/{get_effective_key(args.rna_embedding_key, args.n_HVGs)}_{args.arch}_{args.loss_name}_split{args.split_idx}.json"
        with open(out_path, "w") as f:
            json.dump({
                "rna_embedding_key": get_effective_key(args.rna_embedding_key, args.n_HVGs),
                "arch": args.arch,
                "loss_name": args.loss_name,
                "split_idx": args.split_idx,
                "best_params": best_params,
                "best_value": float(best_row["val_loss"]),
            }, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--rna_embedding_key", required=True, type=str)
    parser.add_argument("--arch", required=True, type=str, choices=["Linear", "MLP", "ResidualMLP"])
    parser.add_argument("--loss_name", required=True, type=str, choices=["Cox", "CIndex"])
    parser.add_argument("--n_HVGs", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_idx", type=int, required=True, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    main(args)
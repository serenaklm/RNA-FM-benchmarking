"""
train/tune_ICI.py

Optuna hyperparameter tuning for ICI response prediction models.

Tunes on IMVigor210 within-cohort splits (5 splits) — the largest ICI cohort,
used as the reference dataset analogous to TCGA in the PFS benchmarking.

Tuned HPs: lr, weight_decay, penalty_weight, dropout_rate (ResidualMLP only).
The same HP ranges as the PFS benchmarking are used for comparability.

Output:
    hp_results_ICI/tuning_history_ICI.csv       — all trial results (append-safe)
    hp_results_ICI/{key}_{arch}_split{N}.json   — per-run best params
    hp_results_ICI/all_aggregated_hps_ICI.json  — run agg_best_hps_ICI.py after all jobs finish

Usage (called by tune_ICI_models.sh, but can also be run directly):
    python -m train.tune_ICI \\
        --rna_embedding_key BulkFormer --n_HVGs 0 \\
        --arch ResidualMLP --split_idx 0 \\
        --n_trials 20
"""

import os
import csv
import fcntl
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import optuna
import torch

torch.set_float32_matmul_precision("high")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pytorch_modules.data.ici_data import ICIResponseDataModule
from pytorch_modules.models.survival_models import RNA_ResponseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
ICI_PKL    = str(REPO_ROOT / "data" / "RNA_embeddings" / "ICI" / "ICI_all_processed.pkl")
SPLIT_BASE = str(REPO_ROOT / "splits" / "ICI" / "within_cohort" / "IMVigor210")
HP_DIR     = REPO_ROOT / "hp_results_ICI"
CSV_PATH   = HP_DIR / "tuning_history_ICI.csv"

FIELDNAMES = [
    "rna_embedding_key", "arch", "split_idx",
    "trial_number", "val_loss",
    "lr", "weight_decay", "penalty_weight", "dropout_rate",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def effective_key(rna_key: str, n_HVGs) -> str:
    """Match the key format used in all_aggregated_hps_ICI.json."""
    if rna_key == "RNA_seq" and n_HVGs is not None and n_HVGs > 0:
        return f"RNA_seq_{n_HVGs}"
    return rna_key


def infer_rna_dim(dm: ICIResponseDataModule) -> int:
    batch = next(iter(dm.train_dataloader()))
    rna = batch.get("x_rna", batch.get("RNA_seq"))
    if rna is None:
        raise KeyError("Batch missing RNA tensor under keys x_rna or RNA_seq.")
    return int(rna.shape[-1])


def compute_pos_weight(dm: ICIResponseDataModule) -> float:
    """Compute n_negative / n_positive from training split for BCE weighting."""
    labels = []
    for batch in dm.train_dataloader():
        labels.extend(batch["y"].cpu().tolist())
    labels = np.array(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def get_completed_trials(eff_key: str, arch: str, split_idx: int) -> int:
    if not CSV_PATH.exists():
        return 0
    try:
        df = pd.read_csv(CSV_PATH)
        return int(df[
            (df["rna_embedding_key"] == eff_key) &
            (df["arch"] == arch) &
            (df["split_idx"] == split_idx)
        ].shape[0])
    except Exception:
        return 0


def append_trial_row(row: dict):
    HP_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(
    args,
    hp_overrides: dict,
    rna_dim: int,
    pos_weight: float,
) -> float:
    """One Optuna trial: train for up to max_epochs with early stopping, return best val loss."""
    pl.seed_everything(args.seed, workers=True)

    dm = ICIResponseDataModule(
        data_path=ICI_PKL,
        split_folder=SPLIT_BASE,
        split_idx=args.split_idx,
        rna_embedding_key=args.rna_embedding_key,
        n_HVGs=args.n_HVGs if args.n_HVGs > 0 else None,
        batch_size=32,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup()

    model = RNA_ResponseModel(
        arch=args.arch,
        rna_dim=rna_dim,
        hidden_dim=512,
        num_layers=3,
        non_linearity="ReLU",
        pos_weight=pos_weight,
        **hp_overrides,
    )

    early_stop = EarlyStopping(monitor="val/loss", patience=20, mode="min")

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.5,
        accumulate_grad_batches=1,
        callbacks=[early_stop],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=dm)
    return float(early_stop.best_score.item())


# ── Optuna objective and callbacks ────────────────────────────────────────────

def make_objective(args, rna_dim: int, pos_weight: float):
    def objective(trial):
        hp_overrides = {
            "lr":             trial.suggest_float("lr",             1e-5, 1e-3, log=True),
            "weight_decay":   trial.suggest_float("weight_decay",   1e-7, 1e-4, log=True),
            "penalty_weight": trial.suggest_float("penalty_weight", 1e-4, 1e-2, log=True),
        }
        if args.arch != "Linear":
            hp_overrides["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)
        else:
            hp_overrides["dropout_rate"] = 0.0

        return run_trial(args, hp_overrides, rna_dim, pos_weight)

    return objective


def make_trial_callback(eff_key: str, args, wandb_run):
    def _cb(study, trial):
        print(
            f"  [{eff_key} | {args.arch} | split {args.split_idx}] "
            f"Trial {trial.number:>3}  val_loss={trial.value:.4f}  "
            f"best={study.best_value:.4f}  {trial.params}"
        )
        append_trial_row({
            "rna_embedding_key": eff_key,
            "arch":              args.arch,
            "split_idx":         args.split_idx,
            "trial_number":      trial.number,
            "val_loss":          trial.value,
            "lr":                trial.params.get("lr",             ""),
            "weight_decay":      trial.params.get("weight_decay",   ""),
            "penalty_weight":    trial.params.get("penalty_weight", ""),
            "dropout_rate":      trial.params.get("dropout_rate",   ""),
        })
        if wandb_run is not None:
            wandb_run.log({
                "trial":          trial.number,
                "val_loss":       trial.value,
                "best_val_loss":  study.best_value,
                **trial.params,
            })
    return _cb


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    eff_key   = effective_key(args.rna_embedding_key, args.n_HVGs)
    run_label = f"{eff_key}_{args.arch}_split{args.split_idx}"

    completed = get_completed_trials(eff_key, args.arch, args.split_idx)
    remaining = args.n_trials - completed

    if remaining <= 0:
        print(f"SKIP — already {completed} trials: {run_label}")
        return

    print(f"\n=== Tuning {run_label}  ({completed} done, {remaining} to go) ===")

    # Build data module once to infer dim and pos_weight
    dm = ICIResponseDataModule(
        data_path=ICI_PKL,
        split_folder=SPLIT_BASE,
        split_idx=args.split_idx,
        rna_embedding_key=args.rna_embedding_key,
        n_HVGs=args.n_HVGs if args.n_HVGs > 0 else None,
        batch_size=32,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup()
    rna_dim    = infer_rna_dim(dm)
    pos_weight = compute_pos_weight(dm)
    print(f"  rna_dim={rna_dim}  pos_weight={pos_weight:.3f}")
    del dm  # free memory; each trial builds its own

    # WandB (optional)
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="ICI-HP-tuning",
                entity=args.user,
                name=run_label,
                config={
                    "rna_embedding_key": eff_key,
                    "arch":              args.arch,
                    "split_idx":         args.split_idx,
                    "n_trials":          args.n_trials,
                    "rna_dim":           rna_dim,
                    "pos_weight":        pos_weight,
                },
                reinit=True,
            )
        except Exception as exc:
            print(f"  WandB init failed ({exc}), continuing without logging.")

    # Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(args, rna_dim, pos_weight),
        n_trials=remaining,
        callbacks=[make_trial_callback(eff_key, args, wandb_run)],
        show_progress_bar=False,
    )

    best = study.best_params
    print(f"  Best val_loss: {study.best_value:.4f}")
    print(f"  Best params:   {best}")

    # Save per-run JSON
    HP_DIR.mkdir(parents=True, exist_ok=True)
    out_json = HP_DIR / f"{run_label}.json"
    with open(out_json, "w") as f:
        json.dump({
            "rna_embedding_key": eff_key,
            "arch":       args.arch,
            "split_idx":  args.split_idx,
            "best_params": best,
            "best_value":  study.best_value,
        }, f, indent=2)
    print(f"  Saved → {out_json}")

    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = study.best_value
        for k, v in best.items():
            wandb_run.summary[f"best_{k}"] = v
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna HP tuning for ICI response prediction (IMVigor210 within-cohort)."
    )
    parser.add_argument("--rna_embedding_key", required=True,
                        help="Embedding key, e.g. BulkFormer, RNA_seq, ssGSEA_immuno, PCA_100")
    parser.add_argument("--n_HVGs",    type=int, default=0,
                        help="Number of HVGs; 0 means no HVG filter (pass None to DataModule).")
    parser.add_argument("--arch",      required=True, choices=["Linear", "ResidualMLP"])
    parser.add_argument("--split_idx", required=True, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--n_trials",  type=int, default=50)
    parser.add_argument("--seed",      type=int, default=37)
    parser.add_argument("--no_wandb",  action="store_true",
                        help="Disable WandB logging (useful for quick local tests).")
    parser.add_argument("--user",      type=str, default=None,
                        help="WandB user/entity ID (default: your WandB default entity).")
    args = parser.parse_args()
    main(args)

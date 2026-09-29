"""
train/train_ICI_model.py

Training script for ICI response prediction (responder vs non-responder).

Supports all three evaluation paradigms:
  within_cohort  — within a single cohort
  loco           — leave-one-cohort-out
  transfer       — cohort-to-cohort transfer

The paradigm is encoded entirely in --split_folder and --cohort_tag; the model
training code is paradigm-agnostic.

Usage examples (run from repo root):

  # Within-cohort (Hugo)
  python -m train.train_ICI_model \
    --dataset ICI_merged --paradigm within_cohort --cohort_tag within_Hugo \
    --split_folder splits/ICI/within_cohort/Hugo \
    --rna_embedding_key RNA_seq --n_HVGs 1250 --arch ResidualMLP --seed 37 --split_idx 0

  # LOCO (held-out = Riaz)
  python -m train.train_ICI_model \
    --dataset ICI_merged --paradigm loco --cohort_tag loco_held_out_Riaz \
    --split_folder splits/ICI/loco/held_out_Riaz \
    --rna_embedding_key BulkFormer --arch Linear --seed 37 --split_idx 0

  # Transfer (Hugo → Rose)
  python -m train.train_ICI_model \
    --dataset ICI_merged --paradigm transfer --cohort_tag transfer_Hugo_to_Rose \
    --split_folder splits/ICI/transfer/Hugo_to_Rose \
    --rna_embedding_key COMPASS --arch MLP --seed 42 --split_idx 0
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
import torch
torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_modules.utils.utils import load_config, write_json, pickle_data
from pytorch_modules.data.ici_data import ICIResponseDataModule
from pytorch_modules.models.survival_models import RNA_ResponseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
ICI_CONFIG  = REPO_ROOT / "configs" / "immunotherapy_cohorts.yaml"
ICI_PKL_DIR = REPO_ROOT / "data" / "RNA_embeddings" / "ICI"

# ── Helpers ───────────────────────────────────────────────────────────────────

@rank_zero_only
def create_results_dir(results_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)


@rank_zero_only
def write_config_wandb(wandb_logger, config):
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy="now")


@rank_zero_only
def write_config_local(config, path):
    write_json(config, path)


def infer_rna_dim(data_module):
    dl = data_module.train_dataloader()
    batch = next(iter(dl))
    rna = batch.get("x_rna", batch.get("RNA_seq"))
    if rna is None:
        raise KeyError("Batch missing RNA tensor under x_rna or RNA_seq.")
    return int(rna.shape[-1])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Config ────────────────────────────────────────────────────────────
    base_cfg = load_config(args.config_path)
    base_cfg["seed"]    = args.seed
    base_cfg["dataset"] = args.dataset
    base_cfg["task"]    = "ici_response"
    base_cfg["paradigm"] = args.paradigm
    base_cfg["cohort_tag"] = args.cohort_tag

    pl.seed_everything(args.seed, workers=True)

    # Resolve data path: look in ICI config by dataset name, or treat as direct path
    with open(ICI_CONFIG) as f:
        ici_cfg = yaml.safe_load(f)

    if args.dataset == "ICI_merged":
        data_path = str(ICI_PKL_DIR / "ICI_all_processed.pkl")
    elif args.dataset in ici_cfg["cohorts"]:
        data_path = str(ICI_PKL_DIR / f"{args.dataset}_processed.pkl")
    else:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. "
            f"Must be 'ICI_merged' or one of {list(ici_cfg['cohorts'].keys())}."
        )

    split_folder = args.split_folder

    print(f"Dataset   : {args.dataset}")
    print(f"Paradigm  : {args.paradigm}  tag={args.cohort_tag}")
    print(f"Data path : {data_path}")
    print(f"Splits    : {split_folder}")

    # ── Data module ───────────────────────────────────────────────────────
    data_cfg = dict(base_cfg.get("data", {}))
    data_cfg.update(
        data_path=data_path,
        split_folder=split_folder,
        split_idx=args.split_idx,
        rna_embedding_key=args.rna_embedding_key,
        n_HVGs=args.n_HVGs,
    )
    # remove survival-only key that ICIResponseDataModule doesn't accept
    data_cfg.pop("endpoint", None)
    data_cfg["batch_size"] = 32
    data_cfg["num_workers"] = 0

    base_cfg["data"] = data_cfg

    dm = ICIResponseDataModule(**data_cfg)
    dm.prepare_data()
    dm.setup()
    print("Data module ready.")

    rna_dim = infer_rna_dim(dm)
    print(f"Inferred rna_dim = {rna_dim}")

    # ── Model ─────────────────────────────────────────────────────────────
    arch      = args.arch
    model_cfg = base_cfg["model"].get(arch, {}).copy()
    model_cfg.setdefault("lr",            base_cfg["model"]["lr"])
    model_cfg.setdefault("weight_decay",  base_cfg["model"]["weight_decay"])
    model_cfg.setdefault("hidden_dim",    512)
    model_cfg.setdefault("num_layers",    3)
    model_cfg.setdefault("non_linearity", "ReLU")
    model_cfg.setdefault("dropout_rate",  0.2)
    model_cfg.setdefault("penalty_weight", 0.0)

    # Optional tuned HPs
    if args.hp_path and Path(args.hp_path).exists():
        with open(args.hp_path) as f:
            all_hps = json.load(f)

        # Map embeddings that share tuned HPs to their representative key
        # (mirrors the PFS sharing logic)
        rna_key = args.rna_embedding_key
        if rna_key.startswith("ssGSEA_"):
            hp_key = "ssGSEA_combined"
        elif rna_key.startswith("max_"):
            hp_key = "max_combined"
        elif rna_key in ("PCA_50", "PCA_100"):
            hp_key = "PCA_50"
        elif rna_key == "RNA_seq" and (args.n_HVGs is not None and args.n_HVGs > 0):
            hp_key = "RNA_seq_1250"
        else:
            hp_key = rna_key

        lookup_key = f"{hp_key}_{arch}_BCE"
        if lookup_key in all_hps:
            model_cfg.update(all_hps[lookup_key])
            print(f"Loaded tuned HPs for {lookup_key} (embedding={rna_key}): {all_hps[lookup_key]}")
        else:
            print(f"No tuned HPs found for {lookup_key}, using config defaults.")

    pos_weight = base_cfg["model"].get("pos_weight", None)
    model = RNA_ResponseModel(
        arch=arch,
        rna_dim=rna_dim,
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        non_linearity=model_cfg["non_linearity"],
        dropout_rate=model_cfg["dropout_rate"],
        penalty_weight=model_cfg["penalty_weight"],
        pos_weight=pos_weight,
    )
    print(f"Model: RNA_ResponseModel  arch={arch}")

    # ── Output directory ──────────────────────────────────────────────────
    expt_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name  = (
        f"{arch}_BCE_rnaKey_{args.rna_embedding_key}_nHVGs_{args.n_HVGs}_"
        f"seed_{args.seed}_split_{args.split_idx}_{expt_time}"
    )
    results_root = REPO_ROOT / "model_outputs" / "ICI" / args.paradigm / args.cohort_tag
    results_dir  = results_root / run_name
    create_results_dir(str(results_dir))

    # ── Callbacks ─────────────────────────────────────────────────────────
    monitor = "val/auprc"
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        dirpath=str(results_dir),
        filename="{epoch:03d}-{val_auprc:.5f}",
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
    )
    earlystop_cb = EarlyStopping(monitor=monitor, patience=base_cfg["callbacks"]["patience"], mode="max")

    # ── Strategy ──────────────────────────────────────────────────────────
    strategy = "auto"
    if base_cfg["trainer"]["devices"] > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    # ── Logger ────────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        save_dir=str(results_dir),
        project=f"ICI_{args.paradigm}_FM_benchmarking{args.wandb_project_suffix}",
        entity=args.user,
        log_model=False,
        name=run_name,
    )
    write_config_wandb(wandb_logger, base_cfg)
    write_config_local(base_cfg, str(results_dir / "config.json"))

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer_cfg = {**base_cfg["trainer"], "accumulate_grad_batches": 1}
    trainer = pl.Trainer(
        default_root_dir=str(results_dir),
        strategy=strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, earlystop_cb],
        **trainer_cfg,
    )

    trainer.fit(model, datamodule=dm)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--user", type=str, default=None)
    p.add_argument(
        "--config_path", required=True, type=str,
        help="Base training config (e.g. configs/all_models.yaml).",
    )
    p.add_argument(
        "--dataset", required=True, type=str,
        help="'ICI_merged' for LOCO/transfer, or a cohort name for within-cohort.",
    )
    p.add_argument(
        "--paradigm", required=True,
        choices=["within_cohort", "loco", "transfer"],
        help="Evaluation paradigm (used only for output path naming).",
    )
    p.add_argument(
        "--cohort_tag", required=True, type=str,
        help="Descriptive tag for the output folder, e.g. 'within_Hugo', 'loco_held_out_Riaz'.",
    )
    p.add_argument(
        "--split_folder", required=True, type=str,
        help="Explicit path to the folder containing split_*.pkl files.",
    )
    p.add_argument("--seed",      type=int, required=True)
    p.add_argument("--split_idx", type=int, required=True)
    p.add_argument(
        "--rna_embedding_key", type=str, required=True,
        help="RNA embedding key (e.g. RNA_seq, BulkFormer, COMPASS, PCA_50, ...).",
    )
    p.add_argument("--n_HVGs", type=int, default=None)
    p.add_argument("--wandb_project_suffix", type=str, default="",
                   help="Optional suffix for the wandb project name, e.g. '_v2'.")
    p.add_argument(
        "--arch", required=True, choices=["Linear", "MLP", "ResidualMLP"],
    )
    p.add_argument(
        "--hp_path", type=str,
        default=str(REPO_ROOT / "hp_results_ICI" / "all_aggregated_hps_ICI.json"),
        help="Path to JSON of tuned HPs (default: hp_results_ICI/all_aggregated_hps_ICI.json).",
    )

    args = p.parse_args()
    main(args)

"""Training script for the binned-HVG ablation experiment.

Mirrors train_RNA_model.py but uses BinnedHVGSurvivalDataModule, which
replaces continuous log-expression values with ordinal bin indices before
feeding them to the downstream survival model.  The goal is to test whether
expression discretization (as performed by foundation models) is itself
responsible for degraded prognostic performance.

New arguments (vs. train_RNA_model.py):
    --n_bins         Number of bins for discretization (e.g. 8, 16, 32, 64).
    --bin_strategy   "linear" (equal-width) or "quantile" (equal-frequency).
    --n_HVGs         Overrides the per-dataset default if provided.

The script defaults n_HVGs to the best-performing value per dataset found
in the main benchmarking experiment (TCGA: 10000, Hartwig: 5000,
CPTAC / TARGET_AML: 1250), so results are directly comparable to the
continuous-HVG baseline.

Hyperparameters are loaded from the same tuned-HP JSON used by the main
benchmarking, mapping binned configs to the RNA_seq_1250 key (same
convention used for all HVG sizes in the main experiment).
"""

import argparse
import json
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from pytorch_modules.data.binned_data import BinnedHVGSurvivalDataModule
from pytorch_modules.models.survival_models import RNA_SurvivalModel
from pytorch_modules.utils.utils import load_config, write_json, checkpoint_every_n_train_steps

torch.set_float32_matmul_precision("high")


# Best-performing HVG count per dataset from the main benchmarking
BEST_N_HVGS: dict = {
    "TCGA":       10000,
    "Hartwig":    5000,
    "CPTAC":      1250,
    "TARGET_AML": 1250,
}


def get_hp_lookup_key(arch: str, loss_name: str) -> str:
    """Map a binned-HVG config to the HP key used in the main benchmarking.

    The main experiment always maps HVG models (any n_HVGs) to the
    RNA_seq_1250 tuned-HP entry, so we follow the same convention here
    for a fair comparison.
    """
    return f"RNA_seq_1250_{arch}_{loss_name}"


@rank_zero_only
def create_results_dir(results_dir: str) -> None:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


@rank_zero_only
def write_config(wandb_logger: WandbLogger, config: dict) -> None:
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy="now")


@rank_zero_only
def write_config_local(config: dict, path: str) -> None:
    write_json(config, path)


def infer_rna_dim_from_batch(data_module: pl.LightningDataModule) -> int:
    dl = data_module.train_dataloader()
    batch = next(iter(dl))
    rna = batch.get("x_rna", batch.get("RNA_seq"))
    if rna is None:
        raise KeyError("Batch missing RNA tensor under keys x_rna or RNA_seq.")
    return int(rna.shape[-1])


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)
    config["seed"] = args.seed

    pl.seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    config["dataset"] = dataset_name
    config["task"] = "survival"

    n_hvgs = args.n_HVGs if args.n_HVGs is not None else BEST_N_HVGS[dataset_name]

    print(
        f"Dataset: {dataset_name} | n_HVGs: {n_hvgs} | "
        f"n_bins: {args.n_bins} | strategy: {args.bin_strategy} | "
        f"arch: {args.arch} | loss: {args.loss_name} | "
        f"seed: {args.seed} | split: {args.split_idx}"
    )

    try:
        data_path = config["data_paths"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing data_paths entry for dataset={dataset_name} in YAML.")

    try:
        split_folder = config["split_folders"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing split_folders entry for dataset={dataset_name} in YAML.")

    endpoint = config["data"].get("endpoint", "PFS")

    data_module = BinnedHVGSurvivalDataModule(
        data_path=data_path,
        split_folder=split_folder,
        split_idx=args.split_idx,
        endpoint=endpoint,
        n_HVGs=n_hvgs,
        n_bins=args.n_bins,
        bin_strategy=args.bin_strategy,
        batch_size=config["data"].get("batch_size", 128),
        num_workers=config["data"].get("num_workers", 4),
    )
    data_module.prepare_data()
    data_module.setup()
    print("Step 1: prepared datasets")

    rna_dim = infer_rna_dim_from_batch(data_module)
    print(f"Inferred rna_dim = {rna_dim}")

    arch = args.arch
    loss_name = args.loss_name
    model_tag = f"{arch}_{loss_name}"
    print(f"Using model: {model_tag}")

    model_cfg = config["model"].get(arch, {}).copy()
    model_cfg.setdefault("lr", config["model"]["lr"])
    model_cfg.setdefault("weight_decay", config["model"]["weight_decay"])
    model_cfg.setdefault("hidden_dim", 512)
    model_cfg.setdefault("num_layers", 3)
    model_cfg.setdefault("non_linearity", "ReLU")
    model_cfg.setdefault("dropout_rate", 0.2)
    model_cfg.setdefault("penalty_weight", 0.0)

    if args.hp_path != "Not provided":
        with open(args.hp_path) as f:
            all_hps = json.load(f)
        lookup_key = get_hp_lookup_key(arch, loss_name)
        if lookup_key in all_hps:
            model_cfg.update(all_hps[lookup_key])
            print(f"Loaded tuned HPs for {lookup_key}: {all_hps[lookup_key]}")
        else:
            print(f"No tuned HPs found for {lookup_key}, using config defaults.")

    model_module = RNA_SurvivalModel(
        arch=arch,
        loss_name=loss_name,
        rna_dim=rna_dim,
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        non_linearity=model_cfg["non_linearity"],
        dropout_rate=model_cfg["dropout_rate"],
        penalty_weight=model_cfg["penalty_weight"],
    )
    print(f"Step 2: prepared model ({model_tag})")

    # -----------------------------------------------------------------------
    # Results directory — kept under a separate binned_HVG_ablation subfolder
    # so outputs never collide with the main benchmarking runs.
    # -----------------------------------------------------------------------
    embedding_tag = f"binned_{args.bin_strategy}_{args.n_bins}_nHVGs_{n_hvgs}"
    results_root = os.path.join(
        "./model_outputs", dataset_name, "survival", endpoint, "binned_HVG_ablation"
    )
    expt_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    expt_name = (
        f"{model_tag}_rnaKey_{embedding_tag}_"
        f"seed_{args.seed}_split_{args.split_idx}_{expt_time}"
    )
    results_dir = os.path.join(results_root, expt_name)
    create_results_dir(results_dir)

    monitor = config["callbacks"]["monitor"]
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=results_dir,
        filename="{epoch:03d}-{val_loss:.5f}",
        every_n_train_steps=checkpoint_every_n_train_steps(config, dataset_name),
        save_top_k=1,
        mode="min",
    )
    earlystop_callback = EarlyStopping(
        monitor=monitor, patience=config["callbacks"]["patience"]
    )

    strategy = "auto"
    if config["trainer"]["devices"] > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    wandb_logger = WandbLogger(
        save_dir=results_dir,
        project=f"{dataset_name}_survival_binned_HVG_ablation",
        entity=args.user,
        log_model=False,
        name=expt_name,
    )

    write_config(wandb_logger, config)
    write_config_local(config, os.path.join(results_dir, "config.json"))

    trainer = pl.Trainer(
        default_root_dir=results_dir,
        strategy=strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, earlystop_callback],
        **config["trainer"],
    )
    print("Step 3: prepared trainer")

    trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train survival model on binned HVG embeddings."
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="Weights & Biases user/entity ID.",
    )
    parser.add_argument(
        "--config_path", required=True, type=str,
        help="Path to the shared YAML config (e.g. configs/all_models.yaml).",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_idx", type=int, required=True,
                        help="Split index (0–4).")
    parser.add_argument(
        "--n_HVGs", type=int, default=None,
        help="Number of HVGs.  Defaults to the best per-dataset value from "
             "the main benchmarking (TCGA=10000, Hartwig=5000, CPTAC/TARGET_AML=1250).",
    )
    parser.add_argument(
        "--n_bins", type=int, required=True,
        help="Number of bins for expression discretization (e.g. 8, 16, 32, 64).",
    )
    parser.add_argument(
        "--bin_strategy", type=str, required=True,
        choices=[
            "linear", "quantile",
            "global_linear", "global_quantile",
            "per_sample_linear", "per_sample_quantile",
        ],
        help=(
            "Binning strategy (scale x method, 3x2 design):\n"
            "  Per-gene : linear (equal-width), quantile (equal-frequency)\n"
            "  Global   : global_linear, global_quantile  — same edges for all genes; "
            "mimics BulkRNABert (global_linear, n_bins=64)\n"
            "  Per-sample: per_sample_linear (equal-width within patient), "
            "per_sample_quantile (rank transform within patient); "
            "mimics scGPT (per_sample_quantile, n_bins=51) and "
            "CancerFoundation (per_sample_quantile, n_bins=100). "
            "Per-sample strategies discard between-patient expression differences by design."
        ),
    )
    parser.add_argument(
        "--arch", type=str, required=True,
        choices=["Linear", "ResidualMLP"],
        help="Downstream model architecture.",
    )
    parser.add_argument(
        "--loss_name", type=str, required=True,
        choices=["Cox", "CIndex"],
        help="Survival loss function.",
    )
    parser.add_argument(
        "--hp_path", type=str, default="Not provided",
        help="Path to tuned-HP JSON from the main benchmarking experiment.",
    )

    args = parser.parse_args()
    main(args)

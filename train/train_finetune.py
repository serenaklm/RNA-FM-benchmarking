"""
Train a LoRA fine-tuned transcriptomic foundation model for survival prediction.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m train.train_finetune \\
    --config_path configs/finetune.yaml \\
    --dataset TCGA --fm_name BulkRNABert \\
    --seed 37 --split_idx 0 --loss_name CIndex
"""

import os
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_data_paths = yaml.safe_load(open(REPO_ROOT / "configs" / "data_paths.yaml"))

import torch
torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_modules.utils.utils import load_config, write_json
from pytorch_modules.data.finetune_data import (
    FinetuneTokensSurvivalDataModule,
    scgpt_collate_fn,
    cancerfoundation_collate_fn,
    state_collate_fn,
)
from pytorch_modules.models.finetune_models import FM_REGISTRY


@rank_zero_only
def create_results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)


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


def main(args):
    config = load_config(args.config_path)
    config["seed"] = args.seed

    pl.seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    fm_name = args.fm_name
    loss_name = args.loss_name
    config["dataset"] = dataset_name
    config["fm_name"] = fm_name

    # ---- FM config ----
    if "fm_configs" not in config or fm_name not in config["fm_configs"]:
        raise KeyError(
            f"fm_configs.{fm_name} not found in {args.config_path}. "
            "Add it before training."
        )
    fm_cfg = config["fm_configs"][fm_name]

    # ---- tokens path ----
    tokens_paths = fm_cfg.get("tokens_paths", {})
    if dataset_name not in tokens_paths:
        raise KeyError(
            f"No tokens_path for dataset={dataset_name} under fm_configs.{fm_name}.tokens_paths"
        )
    tokens_path = tokens_paths[dataset_name]

    # ---- data paths ----
    try:
        data_path = config["data_paths"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing data_paths entry for dataset={dataset_name}")
    try:
        split_folder = config["split_folders"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing split_folders entry for dataset={dataset_name}")

    # ---- collate function (scGPT / CancerFoundation need custom binning + padding) ----
    collate_fn = None
    if fm_name == "scGPT":
        from functools import partial
        import torchtext
        torchtext.disable_torchtext_deprecation_warning()
        from scgpt.tokenizer import GeneVocab
        vocab = GeneVocab.from_file(os.path.join(fm_cfg["hf_model_id"], "vocab.json"))
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in vocab:
                vocab.append_token(s)
        collate_fn = partial(
            scgpt_collate_fn,
            pad_token_id=vocab[fm_cfg.get("pad_token", "<pad>")],
            pad_value=fm_cfg.get("pad_value", -2),
            max_length=fm_cfg.get("max_length", 1200),
            n_bins=fm_cfg.get("n_bins", 51),
        )
    elif fm_name == "CancerFoundation":
        import sys
        from functools import partial
        CF_DIR = _data_paths["model_repos"]["CancerFoundation"]
        sys.path.insert(0, CF_DIR)
        from model.vocab import GeneVocab as CFGeneVocab
        vocab_file = os.path.join(CF_DIR, "model/assets/vocab.json")
        cf_vocab = CFGeneVocab.from_file(vocab_file)
        cf_vocab.set_default_index(cf_vocab["<pad>"])
        collate_fn = partial(
            cancerfoundation_collate_fn,
            pad_token_id=cf_vocab[fm_cfg.get("pad_token", "<pad>")],
            pad_value=fm_cfg.get("pad_value", -2),
            max_length=fm_cfg.get("max_length", 1200),
            n_bins=fm_cfg.get("n_bins", 51),
        )
    elif fm_name == "STATE":
        collate_fn = state_collate_fn

    # ---- data module ----
    data_cfg = dict(config["data"])
    # FM-specific batch_size override (e.g. BulkFormer needs smaller batches)
    effective_batch_size = fm_cfg.get("batch_size", data_cfg.get("batch_size", 4))
    data_module = FinetuneTokensSurvivalDataModule(
        data_path=data_path,
        tokens_path=tokens_path,
        split_folder=split_folder,
        split_idx=args.split_idx,
        endpoint=data_cfg.get("endpoint", "PFS"),
        batch_size=effective_batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )
    data_module.prepare_data()
    data_module.setup()
    print("Step 1: prepared datasets")

    # ---- model ----
    if fm_name not in FM_REGISTRY:
        raise ValueError(
            f"Unknown fm_name='{fm_name}'. Available: {list(FM_REGISTRY.keys())}"
        )
    ModelClass = FM_REGISTRY[fm_name]

    model_kwargs = dict(
        hf_model_id=fm_cfg["hf_model_id"],
        embed_dim=fm_cfg["embed_dim"],
        lora_r=fm_cfg.get("lora_r", 8),
        lora_alpha=fm_cfg.get("lora_alpha", 16),
        lora_dropout=fm_cfg.get("lora_dropout", 0.05),
        target_modules=fm_cfg.get("target_modules", []),
        loss_name=loss_name,
        lr=fm_cfg.get("lr", 5e-4),
        weight_decay=fm_cfg.get("weight_decay", 5e-6),
        penalty_weight=fm_cfg.get("penalty_weight", 0.0),
        use_gradient_checkpointing=fm_cfg.get("use_gradient_checkpointing", False),
        ft_mode=fm_cfg.get("ft_mode", "FFT"),
    )
    if fm_name == "STATE":
        model_kwargs["lora_n_layers"] = fm_cfg.get("lora_n_layers", -1)
    model_module = ModelClass(**model_kwargs)
    print(f"Step 2: prepared {fm_name} model with LoRA (loss={loss_name})")

    # ---- results directory ----
    endpoint = data_cfg.get("endpoint", "PFS")
    ft_mode_label = fm_cfg.get("ft_mode", "LoRA")

    if args.ckpt_path:
        # Resume: reuse the directory that contains the checkpoint so new
        # checkpoints land in the same place and eval_finetune.py finds them.
        results_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        expt_name = os.path.basename(results_dir)
        print(f"Resuming from checkpoint: {args.ckpt_path}")
        print(f"Results dir (reused):     {results_dir}")
    else:
        results_root = os.path.join("./model_outputs_finetune", dataset_name, "survival", endpoint)
        expt_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        expt_name = (
            f"finetune_{fm_name}_{loss_name}_{ft_mode_label}_seed_{args.seed}_split_{args.split_idx}_{expt_time}"
        )
        results_dir = os.path.join(results_root, expt_name)
        create_results_dir(results_dir)

        # Persist config to disk so eval_finetune.py can reload it
        config["data"]["data_path"] = data_path
        config["data"]["tokens_path"] = tokens_path
        config["data"]["split_folder"] = split_folder
        config["data"]["split_idx"] = args.split_idx
        write_config_local(config, os.path.join(results_dir, "config.json"))

    # ---- callbacks ----
    monitor = config["callbacks"]["monitor"]
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=results_dir,
        filename="{epoch:03d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min",
    )
    earlystop_callback = EarlyStopping(
        monitor=monitor, patience=config["callbacks"]["patience"]
    )

    # ---- logger ----
    # When resuming, re-use the existing WandB run so the full training curve
    # appears in a single run rather than being split across two.
    wandb_run_id = None
    if args.ckpt_path:
        import glob as _glob
        wandb_dirs = sorted(_glob.glob(os.path.join(results_dir, "wandb", "run-*")))
        if wandb_dirs:
            # dir name format: run-YYYYMMDD_HHMMSS-<run_id>
            wandb_run_id = os.path.basename(wandb_dirs[-1]).split("-")[-1]
            print(f"Resuming WandB run: {wandb_run_id}")

    wandb_logger = WandbLogger(
        save_dir=results_dir,
        project=f"{dataset_name}_survival_finetune",
        entity=args.user,
        log_model=False,
        name=expt_name,
        id=wandb_run_id,                          # None → new run; str → resume existing
        resume="allow" if wandb_run_id else None,
    )
    write_config_wandb(wandb_logger, config)

    # ---- strategy ----
    strategy = "auto"
    trainer_cfg = dict(config["trainer"])
    if trainer_cfg.get("devices", 1) > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    # ---- trainer ----
    # FM-specific accumulate_grad_batches override (e.g. BulkFormer uses smaller
    # batches so we accumulate more steps to keep the effective batch size ≈ 128)
    if "accumulate_grad_batches" in fm_cfg:
        trainer_cfg["accumulate_grad_batches"] = fm_cfg["accumulate_grad_batches"]
    if "gradient_clip_val" in fm_cfg:
        trainer_cfg["gradient_clip_val"] = fm_cfg["gradient_clip_val"]
    trainer = pl.Trainer(
        default_root_dir=results_dir,
        strategy=strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, earlystop_callback],
        **trainer_cfg,
    )
    print("Step 3: prepared trainer")

    trainer.fit(model_module, datamodule=data_module, ckpt_path=args.ckpt_path or None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="ashyamal-mit")
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["TCGA", "CPTAC", "CALGB", "Hartwig", "ICGC", "TARGET_AML"],
    )
    parser.add_argument(
        "--fm_name",
        required=True,
        help="Foundation model name (must be a key in FM_REGISTRY and fm_configs).",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_idx", type=int, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a Lightning checkpoint to resume from. "
             "The checkpoint's parent directory is reused as the results dir.",
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        required=True,
        choices=["Cox", "CIndex"],
    )
    args = parser.parse_args()
    main(args)

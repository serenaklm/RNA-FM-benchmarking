"""
train_RNA_model_HP_tune.py — helper module for HP tuning.
Provides run_trial() which is called by tune.py for each Optuna trial.
For actual benchmark training, use train_RNA_model.py.
"""

import torch
torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_modules.data.data import (
    TCGADataModule,
    CPTACDataModule,
    CALGBDataModule,
)
from pytorch_modules.models.survival_models import RNA_SurvivalModel


def infer_rna_dim_from_batch(data_module):
    dl = data_module.train_dataloader()
    batch = next(iter(dl))
    rna = batch.get("x_rna", batch.get("RNA_seq"))
    if rna is None:
        raise KeyError("Batch is missing RNA tensor under keys x_rna or RNA_seq.")
    return int(rna.shape[-1])


def _build_data_module(config, args):
    dataset_name = config["dataset"]
    config["data"]["n_HVGs"] = args.n_HVGs
    config["data"]["split_idx"] = args.split_idx
    config["data"]["rna_embedding_key"] = args.rna_embedding_key

    if dataset_name == "TCGA":
        data_module = TCGADataModule(**config["data"])
    elif dataset_name == "CPTAC":
        data_module = CPTACDataModule(**config["data"])
    elif dataset_name == "CALGB":
        data_module = CALGBDataModule(**config["data"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_module.prepare_data()
    data_module.setup()
    return data_module

'''
def _build_model(config, args, rna_dim, hp_overrides=None):
    arch = args.arch
    loss_name = args.loss_name

    model_cfg = config["model"].get(arch, {}).copy()
    # Only lr and weight_decay live at top level; everything else comes from config["model"][arch]
    model_cfg.setdefault("lr", config["model"]["lr"])
    model_cfg.setdefault("weight_decay", config["model"]["weight_decay"])

    if hp_overrides:
        model_cfg.update(hp_overrides)

    return RNA_SurvivalModel(
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

'''

def _build_model(config, args, rna_dim, hp_overrides=None):
    arch = args.arch
    loss_name = args.loss_name

    model_cfg = config["model"].get(arch, {}).copy()
    model_cfg.setdefault("lr", config["model"]["lr"])
    model_cfg.setdefault("weight_decay", config["model"]["weight_decay"])

    if hp_overrides:
        model_cfg.update(hp_overrides)

    return RNA_SurvivalModel(
        arch=arch,
        loss_name=loss_name,
        rna_dim=rna_dim,
        lr=model_cfg["lr"],
        weight_decay=model_cfg["weight_decay"],
        hidden_dim=model_cfg.get("hidden_dim", 512),        # default for Linear
        num_layers=model_cfg.get("num_layers", 3),          # default for Linear
        non_linearity=model_cfg.get("non_linearity", "GELU"), # default for Linear
        dropout_rate=model_cfg.get("dropout_rate", 0.0),    # default for Linear
        penalty_weight=model_cfg["penalty_weight"],
    )
    
def run_trial(config, args, hp_overrides):
    """
    Stripped-down training for Optuna — no wandb, no checkpointing.
    hp_overrides: dict of HPs suggested by Optuna for this trial.
    Returns best val loss (lower is better).
    """
    pl.seed_everything(args.seed, workers=True)

    data_module = _build_data_module(config, args)
    rna_dim = infer_rna_dim_from_batch(data_module)
    model_module = _build_model(config, args, rna_dim, hp_overrides=hp_overrides)

    monitor = config["callbacks"]["monitor"]
    earlystop_callback = EarlyStopping(
        monitor=monitor,
        patience=config["callbacks"]["patience"]
    )

    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="gpu",
        devices=1,
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        callbacks=[earlystop_callback],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model_module, datamodule=data_module)

    return earlystop_callback.best_score.item()
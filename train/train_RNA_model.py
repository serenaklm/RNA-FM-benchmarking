import os
import yaml
import argparse
import json
from datetime import datetime

import torch
torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.trainer.states import TrainerStatus

from pytorch_modules.utils.utils import load_config, write_json, pickle_data, checkpoint_every_n_train_steps

from pytorch_modules.data.data import (
    TCGADataModule,
    CPTACDataModule,
    CALGBDataModule,
    HartwigDataModule,
    HartwigResponseDataModule,
    ICGCDataModule,
    TARGET_AMLDataModule,
)
from pytorch_modules.models.survival_models import (
    RNA_SurvivalModel,
    RNA_ResponseModel
)


def get_effective_key(rna_embedding_key, n_HVGs):
    if rna_embedding_key == "RNA_seq":
        if n_HVGs is None:
            return f"RNA_seq_17490"
        else:
            return "RNA_seq_1250"
    if rna_embedding_key.startswith("ssGSEA_"):
        return "ssGSEA_combined"
    if rna_embedding_key.startswith("max_"):
        return "max_combined"
    if rna_embedding_key.startswith("PCA_"):
        return "PCA_50"
    if rna_embedding_key.startswith("BulkRNABert_HVG_"):
        return "BulkRNABert"
    if rna_embedding_key.startswith("BulkFormer_HVG_"):
        return "BulkFormer"
    return rna_embedding_key

@rank_zero_only
def create_results_dir(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


@rank_zero_only
def write_config(wandb_logger, config):
    run_out_dir = wandb_logger.experiment.dir
    config_out_path = os.path.join(run_out_dir, "run.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(config, f)
    wandb_logger.experiment.save("run.yaml", policy="now")


@rank_zero_only
def write_config_local(config, config_out_path):
    write_json(config, config_out_path)


@rank_zero_only
def write_test_data(test_data, test_data_path):
    pickle_data(test_data, test_data_path)


def infer_rna_dim_from_batch(data_module):
    """Infer rna_dim from one train batch (robust across representations)."""
    dl = data_module.train_dataloader()
    batch = next(iter(dl))

    rna = batch.get("x_rna", batch.get("RNA_seq"))
    if rna is None:
        raise KeyError("Batch is missing RNA tensor under keys x_rna or RNA_seq.")
    rna_dim = int(rna.shape[-1])
    return rna_dim


def main(args):
    # Load config
    config = load_config(args.config_path)
    config["seed"] = args.seed

    # Seed everything
    pl.seed_everything(args.seed, workers=True)
    
    # Get dataset/task
    dataset_name = args.dataset
    task = args.task
    config["dataset"] = dataset_name
    config["task"] = task
    
    print(f"Using dataset: {dataset_name}")
    print(f"Task: {task}")

    # Pull dataset-specific paths from YAML
    try:
        config["data"]["data_path"] = config["data_paths"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing data_paths entry for dataset={dataset_name} in YAML.")

    try:
        config["data"]["split_folder"] = config["split_folders"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing split_folders entry for dataset={dataset_name} in YAML.")

    # -----------------------
    # Build data module
    # -----------------------

    # Inject CLI-controlled fields
    config["data"]["n_HVGs"] = args.n_HVGs
    config["data"]["split_idx"] = args.split_idx
    config["data"]["rna_embedding_key"] = args.rna_embedding_key
    
    if task == "survival":
        if dataset_name == "TCGA":
            data_module = TCGADataModule(**config["data"])
        elif dataset_name == "CPTAC":
            data_module = CPTACDataModule(**config["data"])
        elif dataset_name == "CALGB":
            data_module = CALGBDataModule(**config["data"])
        elif dataset_name == "Hartwig":
            data_module = HartwigDataModule(**config["data"])
        elif dataset_name == "ICGC":
            data_module = ICGCDataModule(**config["data"])
        elif dataset_name == "TARGET_AML":
            data_module = TARGET_AMLDataModule(**config["data"])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
    elif task == "response":
        if dataset_name != "Hartwig":
            raise ValueError("Response task currently supported only for Hartwig.")
        
        data_cfg = dict(config["data"])
        if task == "response":
            data_cfg.pop("endpoint", None)
        data_module = HartwigResponseDataModule(**data_cfg)
        
    else:
        raise ValueError(f"Unknown task: {task}")

    data_module.prepare_data()
    data_module.setup()
    print("Step 1: prepared datasets")

    # Infer dims from a real batch
    rna_dim = infer_rna_dim_from_batch(data_module)

    print("Inferred dimensions: ")
    print(f"  rna_dim = {rna_dim}")

    # -----------------------
    # Build model module
    # -----------------------
    arch = args.arch
    loss_name = args.loss_name
    model_tag = f"{arch}_{loss_name}"
    print(f"Using model {model_tag}")

    # Pull arch-specific overrides if you keep them in config["model"][<arch>]
    model_cfg = config["model"].get(arch, {}).copy()

    # Fill required + defaults
    model_cfg.setdefault("lr", config["model"]["lr"])
    model_cfg.setdefault("weight_decay", config["model"]["weight_decay"])
    model_cfg.setdefault("hidden_dim", 512)
    model_cfg.setdefault("num_layers", 3)
    model_cfg.setdefault("non_linearity", "ReLU")
    model_cfg.setdefault("dropout_rate", 0.2)
    model_cfg.setdefault("penalty_weight", 0.0)
    
    # Override with HPs from tuning if available
    
    if args.hp_path != "Not provided": # Only attempt to load if a path is provided
        with open(args.hp_path) as f:
            all_hps = json.load(f)
        lookup_key = f"{get_effective_key(args.rna_embedding_key, args.n_HVGs)}_{arch}_{loss_name}"
        if lookup_key in all_hps:
            model_cfg.update(all_hps[lookup_key])
            print(f"Loaded tuned HPs for {lookup_key}: {all_hps[lookup_key]}")
        else:
            print(f"No tuned HPs for {lookup_key}, using config defaults.")

    if task == "survival":
        model_module = RNA_SurvivalModel(
            arch=arch,
            loss_name=args.loss_name,
            rna_dim=rna_dim,
            lr=model_cfg["lr"],
            weight_decay=model_cfg["weight_decay"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            non_linearity=model_cfg["non_linearity"],
            dropout_rate=model_cfg["dropout_rate"],
            penalty_weight=model_cfg["penalty_weight"],
        )
    else:
        pos_weight = config["model"].get("pos_weight", None)
        model_module = RNA_ResponseModel(
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

    print(f"Step 2: prepared model ({model_tag})")

    # -----------------------
    # Results directory
    # -----------------------
    # If your config["data"] no longer has task_type, you can simplify this path.
    # Keeping your pattern but removing task_type/fusion:
    endpoint = config["data"].get("endpoint", "PFS")

    if task == "survival":
        results_root = os.path.join("./model_outputs", dataset_name, "survival", endpoint)
    else:
        results_root = os.path.join("./model_outputs", dataset_name, "response")

    expt_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    expt_name = (
        f"{model_tag}_rnaKey_{args.rna_embedding_key}_nHVGs_{args.n_HVGs}_"
        f"seed_{args.seed}_split_{args.split_idx}_{expt_time}"
    )
    results_dir = os.path.join(results_root, expt_name)
    create_results_dir(results_dir)

    # -----------------------
    # Callbacks
    # -----------------------
    monitor = config["callbacks"]["monitor"]
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=results_dir,
        filename="{epoch:03d}-{val_loss:.5f}",
        every_n_train_steps=checkpoint_every_n_train_steps(config, dataset_name),
        save_top_k=1,
        mode="min",
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])

    # -----------------------
    # Strategy
    # -----------------------
    strategy = "auto"
    if config["trainer"]["devices"] > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    # -----------------------
    # Logger
    # -----------------------
    if args.no_wandb:
        logger = CSVLogger(save_dir=results_dir, name="", version="")
    else:
        logger = WandbLogger(
            save_dir=results_dir,
            project=f"{dataset_name}_{task}_FM_benchmarking{args.wandb_project_suffix}",
            entity=args.user,
            log_model=False,
            name=expt_name,
        )
        write_config(logger, config)
    write_config_local(config, os.path.join(results_dir, "config.json"))

    # -----------------------
    # Trainer
    # -----------------------
    trainer = pl.Trainer(
        default_root_dir=results_dir,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback, earlystop_callback],
        **config["trainer"],
    )
    print("Step 3: prepared the trainer")

    # Train
    trainer.fit(model_module, datamodule=data_module)

    # Marks the run as finished; runs without it were interrupted (skipped by eval, redone by the launcher).
    # Lightning catches Ctrl-C and returns from fit() normally, so check the final status.
    if trainer.state.status == TrainerStatus.FINISHED and trainer.global_rank == 0:
        open(os.path.join(results_dir, "done"), "w").close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="The wandb entity (default: your wandb default entity)",
    )
    parser.add_argument(
        "--config_path",
        required=True,
        type=str,
        help="Config file to execute",
    )
    parser.add_argument(
        "--dataset", 
        required=True,
        choices=["TCGA","CPTAC","CALGB","Hartwig","ICGC","TARGET_AML"])
    parser.add_argument(
        "--task", 
        required=True, 
        choices=["survival","response"]
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed",
    )
    parser.add_argument(
        "--split_idx",
        type=int,
        required=True,
        help="Split idx (e.g., 0-4).",
    )
    parser.add_argument(
        "--n_HVGs",
        type=int,
        default=None,
        help="Number of genes (only used when rna_embedding_key is RNA_seq / HVG slicing).",
    )
    parser.add_argument(
        "--rna_embedding_key",
        type=str,
        required=True,
        help="Which processed RNA embedding key to use (e.g., RNA_seq, BulkFormer, ssGSEA_hallmark, etc.).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["Linear", "MLP", "ResidualMLP"],
        help='Model architecture: "Linear", "MLP", or "ResidualMLP".',
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        required=True,
        choices=["Cox", "CIndex"],
        help='Survival objective: "Cox" or "CIndex".',
    )
    parser.add_argument(
        "--hp_path",
        type=str,
        required=True,
        default="Not provided",
        help="Path to JSON file containing tuned HPs",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Log to a local CSV in the run folder instead of wandb.",
    )
    parser.add_argument(
        "--wandb_project_suffix",
        type=str,
        default="",
        help="Optional suffix appended to the wandb project name, e.g. '_v2'.",
    )

    args = parser.parse_args()
    main(args)
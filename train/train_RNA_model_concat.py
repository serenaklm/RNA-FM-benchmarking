import os
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
torch.set_float32_matmul_precision("high")

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from pytorch_modules.utils.utils import load_config, load_pickle, write_json, checkpoint_every_n_train_steps
from pytorch_modules.models.survival_models import RNA_SurvivalModel


REPO_ROOT = Path(__file__).resolve().parents[1]

# COMPASS requires filtering to patients with a TCGA ID mapping
COMPASS_MAPPING_PATHS = {
    "CPTAC":   str(REPO_ROOT / "data" / "CPTAC" / "id_to_tcga_mapping.pkl"),
    "Hartwig": str(REPO_ROOT / "data" / "Hartwig" / "id_to_tcga_mapping.pkl"),
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class ConcatSurvivalDataset(Dataset):
    """
    Returns scaled FM + HVG concatenated features with survival labels.
    Scalers are fit on training data externally and passed in.
    """
    def __init__(
        self,
        data: Dict[str, Dict],
        endpoint: str,
        fm_key: str,
        n_hvgs: int,
        fm_scaler: StandardScaler,
        hvg_scaler: StandardScaler,
    ):
        self.data = data
        self.ids = list(data.keys())
        self.endpoint = endpoint
        self.fm_key = fm_key
        self.n_hvgs = n_hvgs
        self.fm_scaler = fm_scaler
        self.hvg_scaler = hvg_scaler

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.ids[idx]
        rec = self.data[pid]

        x_fm  = np.asarray(rec[self.fm_key]).squeeze()
        x_hvg = np.asarray(rec["RNA_seq"]).squeeze()[:self.n_hvgs]

        x_fm  = self.fm_scaler.transform(x_fm.reshape(1, -1))[0].astype(np.float32)
        x_hvg = self.hvg_scaler.transform(x_hvg.reshape(1, -1))[0].astype(np.float32)

        x = np.concatenate([x_fm, x_hvg])

        t = rec[f"{self.endpoint}_time"]
        e = rec[f"{self.endpoint}_event"]

        return {
            "id_":         pid,
            "x_rna":       torch.from_numpy(x),
            "time":        torch.tensor(float(t), dtype=torch.float32),
            "event":       torch.tensor(float(e), dtype=torch.float32),
            "organ":       rec.get("organ", None),
            "cancer_type": rec.get("cancer_type", None),
        }


# ── DataModule ────────────────────────────────────────────────────────────────

class ConcatSurvivalDataModule(pl.LightningDataModule):
    """
    Loads FM + HVG features from a single pickle, applies separate StandardScalers
    (fit on train only), and concatenates [FM | HVG] as the model input.
    """
    def __init__(
        self,
        data_path: str,
        split_folder: str,
        split_idx: int,
        endpoint: str,
        fm_key: str,
        n_hvgs: int,
        compass_filter=None,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.split_folder = split_folder
        self.split_idx = split_idx
        self.endpoint = endpoint
        self.fm_key = fm_key
        self.n_hvgs = n_hvgs
        self.compass_filter = compass_filter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rna_dim: Optional[int] = None  # set in setup()

    def setup(self, stage: Optional[str] = None):
        data: Dict[str, Dict] = load_pickle(self.data_path)
        split = load_pickle(os.path.join(self.split_folder, f"split_{self.split_idx}.pkl"))

        def subset(ids):
            return {p: data[p] for p in ids if p in data}

        train = subset(split["train"])
        val   = subset(split["val"])
        test  = subset(split["test"])

        n_train_raw, n_val_raw, n_test_raw = len(train), len(val), len(test)

        def keep(pid, rec):
            if rec.get("RNA_seq") is None or rec.get(self.fm_key) is None:
                return False
            if (rec.get(f"{self.endpoint}_time") is None
                    or rec.get(f"{self.endpoint}_event") is None):
                return False
            if self.compass_filter is not None:
                return bool(self.compass_filter(pid, rec))
            return True

        train = {p: r for p, r in train.items() if keep(p, r)}
        val   = {p: r for p, r in val.items()   if keep(p, r)}
        test  = {p: r for p, r in test.items()  if keep(p, r)}

        print(
            "Filtered missing labels / representations:\n"
            f"  Train: {n_train_raw} -> {len(train)}  (dropped {n_train_raw - len(train)})\n"
            f"  Val:   {n_val_raw} -> {len(val)}  (dropped {n_val_raw - len(val)})\n"
            f"  Test:  {n_test_raw} -> {len(test)}  (dropped {n_test_raw - len(test)})"
        )

        # Fit scalers on training data only
        train_fm  = np.stack([np.asarray(r[self.fm_key]).squeeze() for r in train.values()])
        train_hvg = np.stack([np.asarray(r["RNA_seq"]).squeeze()[:self.n_hvgs] for r in train.values()])

        fm_scaler  = StandardScaler().fit(train_fm)
        hvg_scaler = StandardScaler().fit(train_hvg)

        fm_dim = train_fm.shape[1]
        self.rna_dim = fm_dim + self.n_hvgs
        print(f"FM key: {self.fm_key}, FM dim: {fm_dim}, HVG dim: {self.n_hvgs}, concat dim: {self.rna_dim}")

        ds_kwargs = dict(
            endpoint=self.endpoint,
            fm_key=self.fm_key,
            n_hvgs=self.n_hvgs,
            fm_scaler=fm_scaler,
            hvg_scaler=hvg_scaler,
        )
        self.train_dataset = ConcatSurvivalDataset(train, **ds_kwargs)
        self.val_dataset   = ConcatSurvivalDataset(val,   **ds_kwargs)
        self.test_dataset  = ConcatSurvivalDataset(test,  **ds_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          drop_last=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          drop_last=False, pin_memory=True)


# ── rank_zero helpers (unchanged from train_RNA_model.py) ─────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    config = load_config(args.config_path)
    config["seed"] = args.seed
    pl.seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    config["dataset"] = dataset_name
    config["task"] = "survival"

    print(f"Dataset: {dataset_name}  FM: {args.fm_embedding_key}  HVGs: {args.n_HVGs}")

    try:
        config["data"]["data_path"] = config["data_paths"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing data_paths entry for dataset={dataset_name} in YAML.")

    try:
        config["data"]["split_folder"] = config["split_folders"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing split_folders entry for dataset={dataset_name} in YAML.")

    # COMPASS requires filtering to patients that have a TCGA mapping
    compass_filter = None
    if args.fm_embedding_key == "COMPASS" and dataset_name in COMPASS_MAPPING_PATHS:
        valid_ids = set(load_pickle(COMPASS_MAPPING_PATHS[dataset_name]).keys())
        compass_filter = lambda pid, rec: pid in valid_ids

    data_module = ConcatSurvivalDataModule(
        data_path=config["data"]["data_path"],
        split_folder=config["data"]["split_folder"],
        split_idx=args.split_idx,
        endpoint=config["data"]["endpoint"],
        fm_key=args.fm_embedding_key,
        n_hvgs=args.n_HVGs,
        compass_filter=compass_filter,
        batch_size=config["data"].get("batch_size", 64),
        num_workers=config["data"].get("num_workers", 4),
    )
    data_module.prepare_data()
    data_module.setup()
    print("Step 1: prepared datasets")

    rna_dim = data_module.rna_dim
    print(f"  rna_dim (FM + HVG concat) = {rna_dim}")

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = args.arch
    loss_name = args.loss_name
    model_tag = f"{arch}_{loss_name}"
    print(f"Using model {model_tag}")

    model_cfg = config["model"].get(arch, {}).copy()
    model_cfg.setdefault("lr", config["model"]["lr"])
    model_cfg.setdefault("weight_decay", config["model"]["weight_decay"])
    model_cfg.setdefault("hidden_dim", 512)
    model_cfg.setdefault("num_layers", 3)
    model_cfg.setdefault("non_linearity", "ReLU")
    model_cfg.setdefault("dropout_rate", 0.2)
    model_cfg.setdefault("penalty_weight", 0.0)

    # HP lookup always uses HVG-tuned HPs (RNA_seq_1250 key, dataset-agnostic)
    hp_lookup_key = f"RNA_seq_1250_{arch}_{loss_name}"
    if args.hp_path != "Not provided":
        with open(args.hp_path) as f:
            all_hps = json.load(f)
        if hp_lookup_key in all_hps:
            model_cfg.update(all_hps[hp_lookup_key])
            print(f"Loaded tuned HPs for {hp_lookup_key}: {all_hps[hp_lookup_key]}")
        else:
            print(f"No tuned HPs for {hp_lookup_key}, using config defaults.")

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

    # ── Results directory ─────────────────────────────────────────────────────
    endpoint = config["data"].get("endpoint", "PFS")
    results_root = os.path.join("./concat_model_outputs", dataset_name, "survival", endpoint)
    expt_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    expt_name = (
        f"{model_tag}_concat_{args.fm_embedding_key}_HVG{args.n_HVGs}_"
        f"seed_{args.seed}_split_{args.split_idx}_{expt_time}"
    )
    results_dir = os.path.join(results_root, expt_name)
    create_results_dir(results_dir)

    # ── Callbacks ─────────────────────────────────────────────────────────────
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

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy = "auto"
    if config["trainer"]["devices"] > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    # ── Logger ────────────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        save_dir=results_dir,
        project=f"{dataset_name}_survival_FM_concat_benchmarking",
        entity=args.user,
        log_model=False,
        name=expt_name,
    )
    write_config(wandb_logger, config)
    write_config_local(config, os.path.join(results_dir, "config.json"))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        default_root_dir=results_dir,
        strategy=strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, earlystop_callback],
        **config["trainer"],
    )
    print("Step 3: prepared the trainer")

    trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--user", type=str, default=None,
                        help="wandb user ID")
    parser.add_argument("--config_path", required=True, type=str,
                        help="Config YAML path")
    parser.add_argument("--dataset", required=True,
                        choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"])
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    parser.add_argument("--split_idx", type=int, required=True,
                        help="Split index (0-4)")
    parser.add_argument("--fm_embedding_key", type=str, required=True,
                        help="FM embedding key (e.g. BulkFormer, scGPT, COMPASS, ...)")
    parser.add_argument("--n_HVGs", type=int, required=True,
                        help="Number of HVGs to concatenate (dataset-specific best count)")
    parser.add_argument("--arch", type=str, required=True,
                        choices=["Linear", "MLP", "ResidualMLP"])
    parser.add_argument("--loss_name", type=str, required=True,
                        choices=["Cox", "CIndex"])
    parser.add_argument("--hp_path", type=str, required=True, default="Not provided",
                        help="Path to JSON file with tuned HPs")

    args = parser.parse_args()
    main(args)

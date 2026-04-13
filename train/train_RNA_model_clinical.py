#!/usr/bin/env python3
"""
train_RNA_model_clinical.py

Concatenates encoded clinical features with a transcriptomic FM (or raw)
embedding, then trains with the same Linear / ResidualMLP + Cox / CIndex
setup used in the standard benchmarking.

Model input: [FM_embedding | clinical_features]

Clinical features are encoded fit-on-train-only:
  continuous : z-scored + __missing binary flag
  ordinal    : integer map + __missing binary flag
  binary     : {0,1} + __missing binary flag  (0.5 = unknown)
  categorical: one-hot with MISSING category; rare cats (< min_cat_freq) → OTHER
"""
import os
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_data_paths = yaml.safe_load(open(REPO_ROOT / "configs" / "data_paths.yaml"))

import pandas as pd
import torch
torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader

from pytorch_modules.utils.utils import load_config, load_pickle, write_json
from pytorch_modules.models.survival_models import RNA_SurvivalModel


# ---------------------------------------------------------------------------
# Clinical encoding infrastructure  (mirrors train_clinical_baseline.py)
# ---------------------------------------------------------------------------
BAD_VALUES = {"unknown", "not reported", "'--", "not applicable", "n/a", "na",
              "none", "", "nan/null", "--"}

STAGE_MAP = {
    'Stage 0': 0, 'Stage 0a': 0, 'Stage 0is': 0, 'Stage IS': 0,
    'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage IB1': 1, 'Stage IC': 1,
    'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage IIC': 2,
    'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
    'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4, 'Stage IVC': 4,
}
T_MAP = {
    'T0': 0, 'Tis': 0,
    'T1': 1, 'T1a': 1, 'T1b': 1, 'T1c': 1, 'T1d': 1, 'T1mi': 1,
    'T2': 2, 'T2a': 2, 'T2b': 2, 'T2c': 2,
    'T3': 3, 'T3a': 3, 'T3b': 3, 'T3c': 3,
    'T4': 4, 'T4a': 4, 'T4b': 4, 'T4c': 4, 'T4d': 4,
}
N_MAP = {
    'N0': 0, 'N0 (i-)': 0, 'N0 (i+)': 0, 'N0 (mol-)': 0, 'N0 (mol+)': 0,
    'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
    'N2': 2, 'N2a': 2, 'N2b': 2, 'N2c': 2,
    'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
}
M_MAP = {
    'M0': 0, 'cM0 (i+)': 0,
    'M1': 1, 'M1a': 1, 'M1b': 1, 'M1c': 1, 'M1d': 1,
}
BOOL_MAP = {
    'yes': 1.0, 'no': 0.0,
    'true': 1.0, 'false': 0.0,
    '1': 1.0, '0': 0.0,
    'present': 1.0, 'absent': 0.0,
}

FEATURE_SPECS = {
    "TCGA": [
        ("categorical", "project.project_id"),
        ("continuous",  "demographic.age_at_index"),
        ("categorical", "demographic.gender"),
        ("categorical", "demographic.race"),
        ("ordinal",     "diagnoses.ajcc_pathologic_stage",      STAGE_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_t",          T_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_n",          N_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_m",          M_MAP),
        ("binary",      "diagnoses.prior_malignancy"),
        ("binary",      "diagnoses.prior_treatment"),
        ("binary",      "diagnoses.synchronous_malignancy"),
        ("categorical", "diagnoses.tissue_or_organ_of_origin"),
        ("categorical", "diagnoses.tumor_grade"),
        ("categorical", "diagnoses.residual_disease"),
        ("categorical", "diagnoses.morphology"),
        ("categorical", "diagnoses.primary_diagnosis"),
    ],
    "CPTAC": [
        ("continuous",  "age_years"),
        ("categorical", "demographic.gender"),
        ("categorical", "demographic.race"),
        ("ordinal",     "diagnoses.ajcc_pathologic_stage",      STAGE_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_t",          T_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_n",          N_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_m",          M_MAP),
        ("categorical", "diagnoses.tissue_or_organ_of_origin"),
        ("categorical", "diagnoses.tumor_grade"),
        ("categorical", "diagnoses.residual_disease"),
        ("categorical", "diagnoses.morphology"),
        ("categorical", "diagnoses.primary_diagnosis"),
    ],
    "TARGET_AML": [
        ("continuous",  "demographic.age_at_index"),
        ("categorical", "demographic.gender"),
        ("categorical", "demographic.race"),
        ("categorical", "demographic.ethnicity"),
    ],
    "Hartwig": [
        ("continuous",  "age_at_biopsy"),
        ("categorical", "gender"),
        ("categorical", "primaryTumorLocation"),
        ("binary",      "hasSystemicPreTreatment"),
        ("binary",      "hasRadiotherapyPreTreatment"),
    ],
}

# Paths to clinical TSVs (absolute; TCGA/TARGET_AML use base-relative paths)
CLINICAL_PATHS = {
    "TCGA":       "{base}/data/TCGA/clinical/clinical.tsv",
    "CPTAC":      _data_paths["external_data"]["CPTAC_clinical_tsv"],
    "Hartwig":    _data_paths["external_data"]["Hartwig_metadata_tsv"],
    "TARGET_AML": "{base}/data/TARGET_AML/clinical/clinical.tsv",
}
CLINICAL_ID_COLS = {
    "TCGA":       "cases.submitter_id",
    "CPTAC":      "cases.case_id",
    "Hartwig":    "sampleId",
    "TARGET_AML": "cases.submitter_id",
}


def clean_val(x):
    if pd.isnull(x):
        return np.nan
    return np.nan if str(x).strip().lower() in BAD_VALUES else x


def clean_df(df):
    return df.apply(lambda col: col.map(clean_val))


def get_clinical_id(sid, dataset_name):
    """Map sample ID to its clinical index key."""
    if dataset_name == "TARGET_AML":
        return "-".join(sid.rsplit("-", 1)[:1])
    return sid


def load_clinical_dedup(dataset_name, base):
    """Load and deduplicate the clinical TSV; return df indexed by clinical ID."""
    path = CLINICAL_PATHS[dataset_name].format(base=base)
    id_col = CLINICAL_ID_COLS[dataset_name]
    sep = "\t"

    if dataset_name == "Hartwig":
        df = pd.read_csv(path, sep=sep)
        df["biopsyDate"]    = pd.to_datetime(df["biopsyDate"])
        df["age_at_biopsy"] = df["biopsyDate"].dt.year - df["birthYear"]
    else:
        df = clean_df(pd.read_csv(path, sep=sep, low_memory=False))

    if dataset_name == "CPTAC":
        df["age_years"] = (
            pd.to_numeric(df.get("diagnoses.age_at_diagnosis"), errors="coerce") / 365.25
        )

    return df.drop_duplicates(subset=id_col).set_index(id_col)


class FeatureEncoder:
    """
    Fits encoding on training data; transforms train/val/test consistently.
    """
    def __init__(self, min_cat_freq=10):
        self.min_cat_freq = min_cat_freq
        self._specs = []
        self.params_ = {}
        self.feature_names_ = []

    def set_specs(self, specs):
        self._specs = specs

    def fit_transform(self, df):
        self.params_ = {}
        self.feature_names_ = []
        return self._encode(df, fitting=True)

    def transform(self, df):
        return self._encode(df, fitting=False)

    def _encode(self, df, fitting):
        parts = []
        for spec in self._specs:
            ftype, col_name = spec[0], spec[1]
            col = (df[col_name] if col_name in df.columns
                   else pd.Series([np.nan] * len(df), name=col_name, index=df.index))
            if ftype == "continuous":
                parts.append(self._enc_continuous(col_name, col, fitting))
            elif ftype == "ordinal":
                parts.append(self._enc_ordinal(col_name, col, spec[2], fitting))
            elif ftype == "binary":
                parts.append(self._enc_binary(col_name, col, fitting))
            elif ftype == "categorical":
                parts.append(self._enc_categorical(col_name, col, fitting))
        return np.hstack(parts) if parts else np.zeros((len(df), 0), dtype=np.float32)

    def _enc_continuous(self, name, col, fitting):
        vals = pd.to_numeric(col.map(clean_val), errors="coerce")
        if fitting:
            med = float(np.nanmedian(vals))
            std = float(np.nanstd(vals))
            self.params_[name] = {"median": med, "std": std if std > 0 else 1.0}
            self.feature_names_ += [name, f"{name}__missing"]
        p       = self.params_[name]
        missing = vals.isna().astype(float).values
        scaled  = ((vals.fillna(p["median"]) - p["median"]) / p["std"]).values
        return np.column_stack([scaled, missing]).astype(np.float32)

    def _enc_ordinal(self, name, col, mapping, fitting):
        cleaned = col.map(clean_val)
        mapped  = cleaned.map(
            lambda x: mapping.get(str(x), np.nan) if not pd.isnull(x) else np.nan)
        if fitting:
            self.params_[name] = {}
            self.feature_names_ += [name, f"{name}__missing"]
        missing = mapped.isna().astype(float).values
        imputed = mapped.fillna(0).values.astype(float)
        return np.column_stack([imputed, missing]).astype(np.float32)

    def _enc_binary(self, name, col, fitting):
        cleaned = col.map(clean_val)
        mapped  = cleaned.map(
            lambda x: BOOL_MAP.get(str(x).strip().lower(), np.nan)
            if not pd.isnull(x) else np.nan)
        if fitting:
            self.params_[name] = {}
            self.feature_names_ += [name, f"{name}__missing"]
        missing = mapped.isna().astype(float).values
        imputed = mapped.fillna(0.5).values.astype(float)
        return np.column_stack([imputed, missing]).astype(np.float32)

    def _enc_categorical(self, name, col, fitting):
        cleaned = col.map(clean_val).fillna("MISSING").astype(str).str.strip()
        if fitting:
            counts = cleaned.value_counts()
            keep   = set(counts[counts >= self.min_cat_freq].index) | {"MISSING"}
            cats   = sorted(keep)
            self.params_[name] = {"cats": cats}
            self.feature_names_ += [f"{name}__{c}" for c in cats]
        cats    = self.params_[name]["cats"]
        cat_idx = {c: i for i, c in enumerate(cats)}
        mapped  = cleaned.map(lambda x: x if x in cat_idx else "MISSING")
        result  = np.zeros((len(col), len(cats)), dtype=np.float32)
        for j, val in enumerate(mapped):
            result[j, cat_idx[val]] = 1.0
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ClinicalConcatSurvivalDataset(Dataset):
    """
    Returns [FM_embedding | clinical_features] with survival labels.
    FM is used raw (no z-scoring) to match the original benchmarking setup.
    Clinical features are pre-encoded by FeatureEncoder (fit on train only).
    """
    def __init__(
        self,
        data: Dict[str, Dict],
        clinical_feats: Dict[str, np.ndarray],
        endpoint: str,
        fm_key: str,
    ):
        self.data           = data
        self.ids            = list(data.keys())
        self.clinical_feats = clinical_feats
        self.endpoint       = endpoint
        self.fm_key         = fm_key

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.ids[idx]
        rec = self.data[pid]

        x_fm   = np.asarray(rec[self.fm_key]).squeeze().astype(np.float32)
        x_clin = self.clinical_feats[pid]
        x      = np.concatenate([x_fm, x_clin])

        return {
            "id_":         pid,
            "x_rna":       torch.from_numpy(x),
            "time":        torch.tensor(float(rec[f"{self.endpoint}_time"]),  dtype=torch.float32),
            "event":       torch.tensor(float(rec[f"{self.endpoint}_event"]), dtype=torch.float32),
            "organ":       rec.get("organ",       None),
            "cancer_type": rec.get("cancer_type", None),
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class ClinicalConcatSurvivalDataModule(pl.LightningDataModule):
    """
    Loads FM embedding from pkl, encodes clinical features (fit on train only),
    and concatenates [FM | clinical] as the model input.
    """
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        split_folder: str,
        split_idx: int,
        endpoint: str,
        fm_key: str,
        clinical_base: str,
        min_cat_freq: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset_name  = dataset_name
        self.data_path     = data_path
        self.split_folder  = split_folder
        self.split_idx     = split_idx
        self.endpoint      = endpoint
        self.fm_key        = fm_key
        self.clinical_base = clinical_base
        self.min_cat_freq  = min_cat_freq
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.rna_dim: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        data  = load_pickle(self.data_path)
        split = load_pickle(os.path.join(self.split_folder, f"split_{self.split_idx}.pkl"))

        def subset(ids):
            return {p: data[p] for p in ids if p in data}

        train = subset(split["train"])
        val   = subset(split["val"])
        test  = subset(split["test"])

        n_raw = (len(train), len(val), len(test))

        def keep(pid, rec):
            if rec.get(self.fm_key) is None:
                return False
            if (rec.get(f"{self.endpoint}_time")  is None
                    or rec.get(f"{self.endpoint}_event") is None):
                return False
            return True

        train = {p: r for p, r in train.items() if keep(p, r)}
        val   = {p: r for p, r in val.items()   if keep(p, r)}
        test  = {p: r for p, r in test.items()  if keep(p, r)}

        print(
            f"Filtered missing labels / FM embeddings:\n"
            f"  Train: {n_raw[0]} -> {len(train)}\n"
            f"  Val:   {n_raw[1]} -> {len(val)}\n"
            f"  Test:  {n_raw[2]} -> {len(test)}"
        )

        # ---- clinical encoding ----
        clinical_dedup = load_clinical_dedup(self.dataset_name, self.clinical_base)
        specs          = FEATURE_SPECS[self.dataset_name]

        def build_clin_df(ids):
            rows = {}
            for sid in ids:
                cid = get_clinical_id(sid, self.dataset_name)
                rows[sid] = (clinical_dedup.loc[cid].to_dict()
                             if cid in clinical_dedup.index else {})
            return pd.DataFrame.from_dict(rows, orient="index")

        encoder = FeatureEncoder(min_cat_freq=self.min_cat_freq)
        encoder.set_specs(specs)

        train_clin_df = build_clin_df(list(train.keys()))
        train_clin    = encoder.fit_transform(train_clin_df)
        val_clin      = encoder.transform(build_clin_df(list(val.keys())))
        test_clin     = encoder.transform(build_clin_df(list(test.keys())))

        clin_dim = train_clin.shape[1]
        print(f"Clinical feature dim: {clin_dim}  ({len(encoder.feature_names_)} features)")

        def to_dict(ids, arr):
            return {pid: arr[i] for i, pid in enumerate(ids)}

        train_clin_d = to_dict(list(train.keys()), train_clin)
        val_clin_d   = to_dict(list(val.keys()),   val_clin)
        test_clin_d  = to_dict(list(test.keys()),  test_clin)

        # ---- infer FM dim from first training sample ----
        fm_dim = int(np.asarray(next(iter(train.values()))[self.fm_key]).squeeze().shape[0])
        self.rna_dim = fm_dim + clin_dim
        print(f"FM dim: {fm_dim}  Clinical dim: {clin_dim}  Total input dim: {self.rna_dim}")

        ds_kwargs = dict(
            endpoint=self.endpoint,
            fm_key=self.fm_key,
        )
        self.train_dataset = ClinicalConcatSurvivalDataset(train, train_clin_d, **ds_kwargs)
        self.val_dataset   = ClinicalConcatSurvivalDataset(val,   val_clin_d,   **ds_kwargs)
        self.test_dataset  = ClinicalConcatSurvivalDataset(test,  test_clin_d,  **ds_kwargs)

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


# ---------------------------------------------------------------------------
# rank_zero helpers
# ---------------------------------------------------------------------------
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


def get_effective_key(rna_embedding_key):
    """Mirror of train_RNA_model.py's key normalisation for HP lookup."""
    if rna_embedding_key == "RNA_seq":
        return "RNA_seq_17490"
    if rna_embedding_key.startswith("ssGSEA_"):
        return "ssGSEA_combined"
    if rna_embedding_key.startswith("max_"):
        return "max_combined"
    if rna_embedding_key.startswith("PCA_"):
        return "PCA_50"
    return rna_embedding_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    config = load_config(args.config_path)
    config["seed"] = args.seed
    pl.seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    config["dataset"] = dataset_name
    config["task"]    = "survival"

    print(f"Dataset: {dataset_name}  FM: {args.rna_embedding_key}  Seed: {args.seed}  Split: {args.split_idx}")

    try:
        config["data"]["data_path"] = config["data_paths"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing data_paths entry for dataset={dataset_name} in YAML.")

    try:
        config["data"]["split_folder"] = config["split_folders"][dataset_name]
    except KeyError:
        raise KeyError(f"Missing split_folders entry for dataset={dataset_name} in YAML.")

    data_module = ClinicalConcatSurvivalDataModule(
        dataset_name  = dataset_name,
        data_path     = config["data"]["data_path"],
        split_folder  = config["data"]["split_folder"],
        split_idx     = args.split_idx,
        endpoint      = config["data"].get("endpoint", "PFS"),
        fm_key        = args.rna_embedding_key,
        clinical_base = args.clinical_base,
        min_cat_freq  = args.min_cat_freq,
        batch_size    = config["data"].get("batch_size", 64),
        num_workers   = config["data"].get("num_workers", 4),
    )
    data_module.prepare_data()
    data_module.setup()
    print("Step 1: prepared datasets")

    rna_dim = data_module.rna_dim
    print(f"  Total input dim (FM + clinical) = {rna_dim}")

    # ---- model ----
    arch       = args.arch
    loss_name  = args.loss_name
    model_tag  = f"{arch}_{loss_name}"
    print(f"Using model {model_tag}")

    model_cfg = config["model"].get(arch, {}).copy()
    model_cfg.setdefault("lr",             config["model"]["lr"])
    model_cfg.setdefault("weight_decay",   config["model"]["weight_decay"])
    model_cfg.setdefault("hidden_dim",     512)
    model_cfg.setdefault("num_layers",     3)
    model_cfg.setdefault("non_linearity",  "ReLU")
    model_cfg.setdefault("dropout_rate",   0.2)
    model_cfg.setdefault("penalty_weight", 0.0)

    if args.hp_path != "Not provided":
        with open(args.hp_path) as f:
            all_hps = json.load(f)
        lookup_key = f"{get_effective_key(args.rna_embedding_key)}_{arch}_{loss_name}"
        if lookup_key in all_hps:
            model_cfg.update(all_hps[lookup_key])
            print(f"Loaded tuned HPs for {lookup_key}: {all_hps[lookup_key]}")
        else:
            print(f"No tuned HPs for {lookup_key}, using config defaults.")

    model_module = RNA_SurvivalModel(
        arch           = arch,
        loss_name      = loss_name,
        rna_dim        = rna_dim,
        lr             = model_cfg["lr"],
        weight_decay   = model_cfg["weight_decay"],
        hidden_dim     = model_cfg["hidden_dim"],
        num_layers     = model_cfg["num_layers"],
        non_linearity  = model_cfg["non_linearity"],
        dropout_rate   = model_cfg["dropout_rate"],
        penalty_weight = model_cfg["penalty_weight"],
    )
    print(f"Step 2: prepared model ({model_tag})")

    # ---- results dir ----
    endpoint     = config["data"].get("endpoint", "PFS")
    results_root = os.path.join("./clinical_concat_model_outputs", dataset_name, "survival", endpoint)
    expt_time    = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    expt_name    = (
        f"{model_tag}_clinical_concat_{args.rna_embedding_key}_"
        f"seed_{args.seed}_split_{args.split_idx}_{expt_time}"
    )
    results_dir = os.path.join(results_root, expt_name)
    create_results_dir(results_dir)

    # ---- callbacks ----
    monitor = config["callbacks"]["monitor"]
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=results_dir,
        filename="{epoch:03d}-{val_loss:.5f}",
        every_n_train_steps=config["trainer"]["log_every_n_steps"],
        save_top_k=1,
        mode="min",
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=config["callbacks"]["patience"])

    # ---- strategy ----
    strategy = "auto"
    if config["trainer"]["devices"] > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    # ---- logger ----
    wandb_logger = WandbLogger(
        save_dir  = results_dir,
        project   = f"{dataset_name}_survival_FM_clinical_concat",
        entity    = args.user,
        log_model = False,
        name      = expt_name,
    )
    write_config(wandb_logger, config)
    write_config_local(config, os.path.join(results_dir, "config.json"))

    # ---- trainer ----
    trainer = pl.Trainer(
        default_root_dir = results_dir,
        strategy         = strategy,
        logger           = wandb_logger,
        callbacks        = [checkpoint_callback, earlystop_callback],
        **config["trainer"],
    )
    print("Step 3: prepared trainer")

    trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--user",         type=str, default="ashyamal-mit")
    parser.add_argument("--config_path",  type=str, required=True)
    parser.add_argument("--dataset",      type=str, required=True,
                        choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"])
    parser.add_argument("--seed",         type=int, required=True)
    parser.add_argument("--split_idx",    type=int, required=True)
    parser.add_argument("--rna_embedding_key", type=str, required=True,
                        help="FM or RNA_seq embedding key")
    parser.add_argument("--arch",         type=str, required=True,
                        choices=["Linear", "MLP", "ResidualMLP"])
    parser.add_argument("--loss_name",    type=str, required=True,
                        choices=["Cox", "CIndex"])
    parser.add_argument("--hp_path",      type=str, required=True, default="Not provided")
    parser.add_argument("--clinical_base", type=str,
                        default=str(REPO_ROOT),
                        help="Base directory for resolving relative clinical TSV paths")
    parser.add_argument("--min_cat_freq", type=int, default=10,
                        help="Min train-set frequency to keep a categorical value")

    args = parser.parse_args()
    main(args)

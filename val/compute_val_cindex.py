# val/compute_val_cindex.py
#
# Computes val C-index for all trained survival models by loading saved checkpoints
# and running inference on the validation split. Used for val-based model selection
# (arch x loss) in the analysis notebook.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 python -m val.compute_val_cindex --dataset TCGA
#   CUDA_VISIBLE_DEVICES=1 python -m val.compute_val_cindex --dataset Hartwig
#   CUDA_VISIBLE_DEVICES=2 python -m val.compute_val_cindex --dataset CPTAC
#   CUDA_VISIBLE_DEVICES=3 python -m val.compute_val_cindex --dataset TARGET_AML
#
# Output: {results_dir}/val_metrics.csv  (one row per run, overall val C-index only)

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

from pytorch_modules.utils.utils import load_json
from pytorch_modules.data.data import (
    TCGADataModule,
    CPTACDataModule,
    CALGBDataModule,
    HartwigDataModule,
    ICGCDataModule,
    TARGET_AMLDataModule,
)
from pytorch_modules.models.survival_models import RNA_SurvivalModel


RUN_RE = re.compile(
    r"^(?P<arch>[^_]+)_(?P<loss>[^_]+)_rnaKey_(?P<rna_key>.+?)_nHVGs_(?P<n_hvgs>[\w]+)_seed_(?P<seed>\d+)_split_(?P<split>\d+)_"
)

SURVIVAL_DM = {
    "TCGA":       TCGADataModule,
    "CPTAC":      CPTACDataModule,
    "CALGB":      CALGBDataModule,
    "Hartwig":    HartwigDataModule,
    "ICGC":       ICGCDataModule,
    "TARGET_AML": TARGET_AMLDataModule,
}


def parse_run_name(folder_name: str):
    m = RUN_RE.search(folder_name)
    if not m:
        return None
    d = m.groupdict()
    d["n_hvgs"] = None if d["n_hvgs"] == "None" else int(d["n_hvgs"])
    d["seed"]   = int(d["seed"])
    d["split"]  = int(d["split"])
    return d


def load_done_keys(out_csv: Path):
    if not out_csv.exists():
        return set(), None
    df = pd.read_csv(out_csv)
    needed = {"arch", "loss", "RNA_key", "nHVGs", "seed", "split"}
    if not needed.issubset(df.columns):
        return set(), df
    done = set(zip(
        df["arch"].astype(str),
        df["loss"].astype(str),
        df["RNA_key"].astype(str),
        df["nHVGs"].apply(lambda x: None if pd.isna(x) else int(x)),
        df["seed"].astype(int),
        df["split"].astype(int),
    ))
    return done, df


def safe_cindex(event_bool, time, risk):
    event_bool = np.asarray(event_bool, dtype=bool).reshape(-1)
    time       = np.asarray(time,       dtype=float).reshape(-1)
    risk       = np.asarray(risk,       dtype=float).reshape(-1)
    if len(risk) < 2 or np.sum(event_bool) == 0:
        return float("nan")
    try:
        cindex, *_ = concordance_index_censored(event_bool, time, risk)
        return float(cindex)
    except Exception:
        return float("nan")


def eval_val_cindex(model_folder: Path, dataset: str, use_gpu: bool) -> float:
    """Load checkpoint, run inference on val split, return overall C-index."""
    config_path = model_folder / "config.json"
    if not config_path.exists():
        return None

    config = load_json(config_path)
    pl.seed_everything(int(config.get("seed", 37)), workers=True)

    dm = SURVIVAL_DM[dataset](**config["data"])
    dm.prepare_data()
    dm.setup()

    ckpts = sorted([f for f in os.listdir(model_folder) if f.endswith(".ckpt")])
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {model_folder}")
    ckpt_path = model_folder / ckpts[0]

    model = RNA_SurvivalModel.load_from_checkpoint(str(ckpt_path))
    model.eval()

    trainer_kwargs = dict(config.get("trainer", {}))
    for key in ("devices", "accelerator", "strategy"):
        trainer_kwargs.pop(key, None)

    accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
    trainer = pl.Trainer(
        default_root_dir=str(model_folder),
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    with torch.no_grad():
        pred_batches = trainer.predict(model, dataloaders=dm.val_dataloader())

    risk_list, time_list, event_list = [], [], []
    for batch in pred_batches:
        _, risk, time, event, *_ = batch
        risk_list.extend(list(risk))
        time_list.extend(list(time))
        event_list.extend(list(event))

    return safe_cindex(event_list, time_list, risk_list)


def main(args):
    dataset    = args.dataset
    base       = Path("/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs") / dataset / "survival" / "PFS"
    results_dir = Path(args.results_dir) if args.results_dir else base
    out_csv     = Path(args.out_csv)     if args.out_csv     else (base / "val_metrics.csv")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"Dataset={dataset} | Found {len(folders)} subfolders in {results_dir}")

    done_keys, existing_df = load_done_keys(out_csv)
    print(f"Loaded {len(done_keys)} existing entries" if existing_df is not None else "No existing CSV found")

    new_rows = []
    skipped  = 0

    for i, folder in tqdm(enumerate(folders, 1), total=len(folders)):
        meta = parse_run_name(folder.name)
        if meta is None:
            continue

        key = (meta["arch"], meta["loss"], meta["rna_key"], meta["n_hvgs"], meta["seed"], meta["split"])
        if key in done_keys:
            skipped += 1
            continue

        row = {
            "arch":    meta["arch"],
            "loss":    meta["loss"],
            "RNA_key": meta["rna_key"],
            "nHVGs":   meta["n_hvgs"],
            "seed":    meta["seed"],
            "split":   meta["split"],
            "val_cindex": np.nan,
            "error":   "",
        }

        try:
            val_c = eval_val_cindex(folder, dataset=dataset, use_gpu=True)
            if val_c is None:
                if args.verbose:
                    print(f"SKIP (no config.json): {folder.name}")
                continue
            row["val_cindex"] = val_c
        except Exception as e:
            row["error"] = str(e)

        new_rows.append(row)
        if args.verbose:
            print(f"[{i}/{len(folders)}] {folder.name} val_cindex={row['val_cindex']:.4f}")

    print(f"Skipped {skipped} already-evaluated runs")

    df_new = pd.DataFrame(new_rows)
    df_out = df_new if existing_df is None else pd.concat([existing_df, df_new], ignore_index=True, sort=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} rows to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True, choices=list(SURVIVAL_DM.keys()))
    p.add_argument("--results_dir", default=None,  help="Folder containing run subfolders (default: model_outputs/{dataset}/survival/PFS)")
    p.add_argument("--out_csv",     default=None,  help="Output CSV path (default: results_dir/val_metrics.csv)")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()
    main(args)

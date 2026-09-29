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
    compass_eligible_ids,
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
    # Rows that failed to score are dropped (and rewritten) so they are retried
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str) == "")].reset_index(drop=True)
    needed = {"arch", "loss", "RNA_key", "nHVGs", "seed", "split"}
    if not needed.issubset(df.columns):
        return set(), df
    done = set(zip(
        df["arch"].astype(str),
        df["loss"].astype(str),
        df["RNA_key"].astype(str),
        [None if pd.isna(x) else int(x) for x in df["nHVGs"]],  # list, not .apply: pandas would turn None into NaN
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


def eval_val_cindex(model_folder: Path, dataset: str, use_gpu: bool, keep_ids=None) -> float:
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
        ids, risk, time, event, *_ = batch
        for pid, r, t, e in zip(ids, risk, time, event):
            if keep_ids is None or pid in keep_ids:
                risk_list.append(r)
                time_list.append(t)
                event_list.append(e)

    return safe_cindex(event_list, time_list, risk_list)



def eval_subset_suffix(args) -> str:
    return ("_compass_subset" if args.compass_subset else "") + ("_clean" if args.eval_ids_file else "")


def resolve_keep_ids(folder: Path, dataset: str, args, cache: dict):
    """Samples to score for this run (None = all). --compass_subset keeps patients COMPASS runs use;
    --eval_ids_file keeps the per-split 'val' IDs listed in the file. Both flags intersect."""
    if not (args.compass_subset or args.eval_ids_file):
        return None
    data_cfg = load_json(folder / "config.json")["data"]
    keep = None
    if args.compass_subset:
        if data_cfg["data_path"] not in cache:
            cache[data_cfg["data_path"]] = compass_eligible_ids(dataset, data_cfg["data_path"])
        keep = cache[data_cfg["data_path"]]
    if args.eval_ids_file:
        if "eval_ids" not in cache:
            cache["eval_ids"] = load_json(args.eval_ids_file)
        listed = set(cache["eval_ids"][str(data_cfg["split_idx"])]["val"])
        keep = listed if keep is None else keep & listed
    return keep

def main(args):
    dataset    = args.dataset
    base       = Path("./model_outputs") / dataset / "survival" / "PFS"
    results_dir = Path(args.results_dir) if args.results_dir else base
    default_csv = f"val_metrics{eval_subset_suffix(args)}.csv"
    out_csv     = Path(args.out_csv)     if args.out_csv     else (base / default_csv)

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    unfinished = [p.name for p in folders if not (p / "done").exists()]
    if unfinished and not args.include_unfinished:
        print(f"Skipping {len(unfinished)} unfinished runs (no 'done' marker), e.g. {unfinished[:3]}")
        folders = [p for p in folders if (p / "done").exists()]
    print(f"Dataset={dataset} | Found {len(folders)} subfolders in {results_dir}")

    done_keys, existing_df = load_done_keys(out_csv)
    print(f"Loaded {len(done_keys)} existing entries" if existing_df is not None else "No existing CSV found")

    new_rows = []
    skipped  = 0
    keep_ids_cache = {}

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
            keep_ids = resolve_keep_ids(folder, dataset, args, keep_ids_cache)
            val_c = eval_val_cindex(folder, dataset=dataset, use_gpu=True, keep_ids=keep_ids)
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
    p.add_argument("--compass_subset", action="store_true",
                   help="Score only patients COMPASS runs use, so every representation is compared on the same validation patients. Default output: val_metrics_compass_subset.csv")
    p.add_argument("--eval_ids_file", default=None,
                   help="JSON {split_idx: {'val': [...], 'test': [...]}} of samples to score (e.g. splits/TARGET_AML/clean_eval_ids.json). Default output gets a '_clean' suffix")
    p.add_argument("--include_unfinished", action="store_true",
                   help="Also score run folders without a 'done' marker (runs trained before markers were written).")
    args = p.parse_args()
    main(args)

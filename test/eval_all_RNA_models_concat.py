# test/eval_all_RNA_models_concat.py
#
# Eval script for FM + HVG concatenated models (incremental value experiment).
# Minimally modified version of test/eval_all_RNA_models.py.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 python -m test.eval_all_RNA_models_concat --dataset TCGA    --endpoint PFS
#   CUDA_VISIBLE_DEVICES=1 python -m test.eval_all_RNA_models_concat --dataset Hartwig  --endpoint PFS
#   CUDA_VISIBLE_DEVICES=2 python -m test.eval_all_RNA_models_concat --dataset CPTAC    --endpoint PFS
#   CUDA_VISIBLE_DEVICES=3 python -m test.eval_all_RNA_models_concat --dataset TARGET_AML --endpoint PFS
#
# Assumes each run folder contains: config.json + one .ckpt
# Outputs to: concat_model_outputs/{dataset}/survival/{endpoint}/all_metrics.csv

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

from pytorch_modules.utils.utils import load_json, load_pickle
from pytorch_modules.models.survival_models import RNA_SurvivalModel
from train.train_RNA_model_concat import ConcatSurvivalDataModule, COMPASS_MAPPING_PATHS

REPO_ROOT = Path(__file__).resolve().parents[1]


# Matches concat run folder names:
#   {arch}_{loss}_concat_{fm_key}_HVG{n_hvgs}_seed_{seed}_split_{split}_{timestamp}
RUN_RE = re.compile(
    r"^(?P<arch>[^_]+)_(?P<loss>[^_]+)_concat_(?P<fm_key>.+?)_HVG(?P<n_hvgs>\d+)_seed_(?P<seed>\d+)_split_(?P<split>\d+)_"
)


def parse_run_name(folder_name: str):
    m = RUN_RE.search(folder_name)
    if not m:
        return None
    d = m.groupdict()
    d["n_hvgs"] = int(d["n_hvgs"])
    d["seed"] = int(d["seed"])
    d["split"] = int(d["split"])
    return d


def load_done_keys(out_csv: Path):
    if not out_csv.exists():
        return set(), None

    df = pd.read_csv(out_csv)
    base_needed = {"arch", "loss", "fm_key", "n_hvgs", "seed", "split"}
    if not base_needed.issubset(df.columns):
        return set(), df

    done = set(
        zip(
            df["arch"].astype(str),
            df["loss"].astype(str),
            df["fm_key"].astype(str),
            df["n_hvgs"].astype(int),
            df["seed"].astype(int),
            df["split"].astype(int),
        )
    )
    return done, df


def safe_cindex(event_bool, time, risk):
    event_bool = np.asarray(event_bool, dtype=bool).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    if len(risk) < 2 or np.sum(event_bool) == 0:
        return float("nan")
    try:
        cindex, _, _, _, _ = concordance_index_censored(event_bool, time, risk)
        return float(cindex)
    except Exception:
        return float("nan")


def _consolidate_survival(pred_batches):
    id_list, risk_list, time_list, event_list, organ_list, ct_list = [], [], [], [], [], []
    for batch in pred_batches:
        id_, risk, time, event, organ, cancer_type = batch
        id_list.extend(list(id_))
        risk_list.extend(list(risk))
        time_list.extend(list(time))
        event_list.extend(list(event))
        organ_list.extend(list(organ))
        ct_list.extend(list(cancer_type))
    return id_list, risk_list, time_list, event_list, organ_list, ct_list


def eval_one_folder(
    model_folder: Path,
    dataset: str,
    meta: dict,
    min_n: int,
    min_events: int,
    use_gpu: bool,
):
    config_path = model_folder / "config.json"
    if not config_path.exists():
        return None

    config = load_json(config_path)
    pl.seed_everything(int(config.get("seed", 37)), workers=True)

    if "data" not in config:
        raise KeyError(f"config.json missing 'data' field: {config_path}")
    data_cfg = config["data"]

    # COMPASS filter
    compass_filter = None
    if meta["fm_key"] == "COMPASS" and dataset in COMPASS_MAPPING_PATHS:
        valid_ids = set(load_pickle(COMPASS_MAPPING_PATHS[dataset]).keys())
        compass_filter = lambda pid, rec: pid in valid_ids

    dm = ConcatSurvivalDataModule(
        data_path=data_cfg["data_path"],
        split_folder=data_cfg["split_folder"],
        split_idx=meta["split"],
        endpoint=data_cfg["endpoint"],
        fm_key=meta["fm_key"],
        n_hvgs=meta["n_hvgs"],
        compass_filter=compass_filter,
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
    )
    dm.prepare_data()
    dm.setup()

    # Checkpoint
    ckpts = sorted([f for f in os.listdir(model_folder) if f.endswith(".ckpt")])
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .ckpt in {model_folder}")
    ckpt_path = model_folder / ckpts[0]

    model = RNA_SurvivalModel.load_from_checkpoint(str(ckpt_path))
    model.eval()

    trainer_kwargs = dict(config.get("trainer", {}))
    trainer_kwargs.pop("devices", None)
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("strategy", None)

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
        pred_batches = trainer.predict(model, dataloaders=dm.test_dataloader())

    _, risk, time, event, organ, cancer_type = _consolidate_survival(pred_batches)

    event_bool = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    cancer_type = list(cancer_type)

    results = {"overall": safe_cindex(event_bool, time, risk)}

    for ct in set(cancer_type):
        if ct is None:
            continue
        idx = [i for i, ci in enumerate(cancer_type) if ci == ct]
        if len(idx) < min_n:
            continue
        if int(np.sum(event_bool[idx])) < min_events:
            continue
        results[str(ct)] = safe_cindex(event_bool[idx], time[idx], risk[idx])

    return results


def main(args):
    dataset = args.dataset
    endpoint = args.endpoint

    base = (
        REPO_ROOT / "concat_model_outputs"
        / dataset / "survival" / endpoint
    )
    results_dir = Path(args.results_dir) if args.results_dir is not None else base
    out_csv = Path(args.out_csv) if args.out_csv is not None else (base / "all_metrics.csv")

    print(f"Dataset={dataset}  endpoint={endpoint}")
    print(f"Results dir: {results_dir}")
    print(f"Output CSV:  {out_csv}")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"Found {len(folders)} subfolders in {results_dir}")

    done_keys, existing_df = load_done_keys(out_csv)
    print(f"Loaded {len(done_keys)} existing entries from {out_csv}" if existing_df is not None else "No existing CSV found")

    new_rows = []
    skipped = 0

    for i, folder in tqdm(enumerate(folders, 1), total=len(folders)):
        meta = parse_run_name(folder.name)
        if meta is None:
            continue

        key = (
            meta["arch"],
            meta["loss"],
            meta["fm_key"],
            meta["n_hvgs"],
            int(meta["seed"]),
            int(meta["split"]),
        )

        if key in done_keys:
            skipped += 1
            if args.verbose:
                print(f"SKIP already in CSV: {key}")
            continue

        try:
            res = eval_one_folder(
                folder,
                dataset=dataset,
                meta=meta,
                min_n=args.min_n,
                min_events=args.min_events,
                use_gpu=True,
            )
        except Exception as e:
            row = {
                "run_dir": str(folder),
                "dataset": dataset,
                "task": "survival",
                "arch": meta["arch"],
                "loss": meta["loss"],
                "fm_key": meta["fm_key"],
                "n_hvgs": meta["n_hvgs"],
                "seed": meta["seed"],
                "split": meta["split"],
                "overall": np.nan,
                "error": str(e),
            }
            new_rows.append(row)
            continue

        if res is None:
            if args.verbose:
                print(f"SKIP (no config.json): {folder.name}")
            continue

        row = {
            "run_dir": str(folder),
            "dataset": dataset,
            "task": "survival",
            "arch": meta["arch"],
            "loss": meta["loss"],
            "fm_key": meta["fm_key"],
            "n_hvgs": meta["n_hvgs"],
            "seed": meta["seed"],
            "split": meta["split"],
            "overall": res.get("overall", np.nan),
            "error": "",
        }
        for k, v in res.items():
            if k != "overall":
                row[k] = v

        new_rows.append(row)
        if args.verbose:
            print(f"[{i}/{len(folders)}] {folder.name}  overall={row['overall']:.4f}")

    print(f"Skipped {skipped} runs already in CSV")

    df_new = pd.DataFrame(new_rows)
    df_out = df_new if existing_df is None else pd.concat([existing_df, df_new], ignore_index=True, sort=False)

    base_cols = ["dataset", "task", "arch", "loss", "fm_key", "n_hvgs", "seed", "split", "overall"]
    extra_cols = ["run_dir", "error"]
    organ_cols = sorted([c for c in df_out.columns if c not in set(base_cols + extra_cols)])

    cols = base_cols + organ_cols + extra_cols
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote CSV with {len(df_out)} rows: {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True, choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"])
    p.add_argument("--endpoint", default="PFS")

    p.add_argument("--results_dir", default=None, help="Folder containing run subfolders")
    p.add_argument("--out_csv", default=None, help="Where to write the CSV")

    p.add_argument("--min_n", type=int, default=0)
    p.add_argument("--min_events", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    main(args)

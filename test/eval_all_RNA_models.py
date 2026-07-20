# eval/eval_all_RNA_models.py
#
# Examples:
#   CUDA_VISIBLE_DEVICES=2 python -m test.eval_all_RNA_models --dataset TCGA   --task survival --endpoint PFS
#   CUDA_VISIBLE_DEVICES=3 python -m test.eval_all_RNA_models --dataset CPTAC  --task survival --endpoint PFS
#   CUDA_VISIBLE_DEVICES=0 python -m test.eval_all_RNA_models --dataset CALGB  --task survival --endpoint PFS
#   CUDA_VISIBLE_DEVICES=0 python -m test.eval_all_RNA_models --dataset Hartwig --task survival --endpoint PFS
#   CUDA_VISIBLE_DEVICES=0 python -m test.eval_all_RNA_models --dataset Hartwig --task response
#
# Optional overrides:
#   python -m test.eval_all_RNA_models \
#     --dataset CPTAC --task survival --endpoint PFS \
#     --results_dir /data/.../model_outputs/CPTAC/survival/PFS \
#     --out_csv     /data/.../model_outputs/CPTAC/survival/PFS/all_metrics.csv
#
# Assumes each run folder contains: config.json + one .ckpt
#
# predict_step expectations:
#   survival: (id_, risk, time, event, organ, cancer_type)
#   response: (id_, score, y_true, organ)

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from pytorch_modules.utils.utils import load_json
from pytorch_modules.data.data import (
    TCGADataModule,
    CPTACDataModule,
    CALGBDataModule,
    HartwigDataModule,
    HartwigResponseDataModule,
    ICGCDataModule,
    TARGET_AMLDataModule,
)
from pytorch_modules.models.survival_models import RNA_SurvivalModel, RNA_ResponseModel


# Matches your new training script run folder name:
#   {arch}_{loss}_rnaKey_{rna_key}_nHVGs_{n}_seed_{seed}_split_{split}_{timestamp}
RUN_RE = re.compile(
    r"^(?P<arch>[^_]+)_(?P<loss>[^_]+)_rnaKey_(?P<rna_key>.+?)_nHVGs_(?P<n_hvgs>[\w]+)_seed_(?P<seed>\d+)_split_(?P<split>\d+)_"
)

SURVIVAL_DM = {
    "TCGA": TCGADataModule,
    "CPTAC": CPTACDataModule,
    "CALGB": CALGBDataModule,
    "Hartwig": HartwigDataModule,
    "ICGC": ICGCDataModule,
    "TARGET_AML": TARGET_AMLDataModule,
}

# response supported only for Hartwig in your training script
RESPONSE_DM = {
    "Hartwig": HartwigResponseDataModule,
}

def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def safe_f1_macro(y_true, y_score, threshold: float = 0.5):
    """
    Macro-F1 for binary response.
    If scores look like logits (outside [0,1]), apply sigmoid.
    Then threshold to hard labels.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)

    if len(y_true) < 2:
        return float("nan")

    # logits -> probs if needed
    if np.nanmin(y_score) < 0.0 or np.nanmax(y_score) > 1.0:
        y_prob = _sigmoid(y_score)
    else:
        y_prob = y_score

    y_pred = (y_prob >= threshold).astype(int)

    try:
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float("nan")

def canon_rna_key(x) -> str:
    return str(x).strip()


def parse_run_name(folder_name: str):
    m = RUN_RE.search(folder_name)
    if not m:
        return None
    d = m.groupdict()
    d["n_hvgs"] = None if d["n_hvgs"] == "None" else int(d["n_hvgs"])
    d["seed"] = int(d["seed"])
    d["split"] = int(d["split"])
    return d


def load_done_keys(out_csv: Path):
    """
    Returns:
      done_keys: set of tuples that identify already-evaluated runs
      existing_df: existing csv dataframe or None
    """
    if not out_csv.exists():
        return set(), None

    df = pd.read_csv(out_csv)

    # We include task in the key if present, to avoid mixing survival/response.
    base_needed = {"arch", "loss", "RNA_key", "nHVGs", "seed", "split"}
    has_task = "task" in df.columns

    if not base_needed.issubset(df.columns):
        # can't safely detect duplicates; return empty done_keys but preserve df
        return set(), df

    if has_task:
        done = set(
            zip(
                df["task"].astype(str),
                df["arch"].astype(str),
                df["loss"].astype(str),
                df["RNA_key"].apply(canon_rna_key),
                df["nHVGs"].apply(lambda x: None if pd.isna(x) else int(x)),
                df["seed"].astype(int),
                df["split"].astype(int),
            )
        )
    else:
        done = set(
            zip(
                df["arch"].astype(str),
                df["loss"].astype(str),
                df["RNA_key"].apply(canon_rna_key),
                df["nHVGs"].apply(lambda x: None if pd.isna(x) else int(x)),
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


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def build_datamodule(dataset: str, task: str, data_cfg: dict):
    cfg = dict(data_cfg)

    if task == "survival":
        if dataset not in SURVIVAL_DM:
            raise ValueError(f"Unknown dataset={dataset} for survival. Choose from {list(SURVIVAL_DM.keys())}")
        return SURVIVAL_DM[dataset](**cfg)

    if task == "response":
        if dataset not in RESPONSE_DM:
            raise ValueError(f"Response task supported only for {list(RESPONSE_DM.keys())}. Got dataset={dataset}.")
        
        cfg.pop("endpoint", None)
        return RESPONSE_DM[dataset](**cfg)

    raise ValueError(f"Unknown task={task}")

'''
def _consolidate_survival(pred_batches):
    id_list, risk_list, time_list, event_list, organ_list = [], [], [], [], []
    for batch in pred_batches:
        id_, risk, time, event, organ = batch
        id_list.extend(list(id_))
        risk_list.extend(list(risk))
        time_list.extend(list(time))
        event_list.extend(list(event))
        organ_list.extend(list(organ))
    return id_list, risk_list, time_list, event_list, organ_list
'''

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

def _consolidate_response(pred_batches):
    id_list, score_list, y_list, organ_list = [], [], [], []
    for batch in pred_batches:
        id_, score, y_true, organ = batch
        id_list.extend(list(id_))
        score_list.extend(list(score))
        y_list.extend(list(y_true))
        organ_list.extend(list(organ))
    return id_list, score_list, y_list, organ_list


def eval_one_folder(
    model_folder: Path,
    dataset: str,
    task: str,
    min_n: int,
    min_events: int,
    use_gpu: bool,
    threshold: float = 0.5
):
    config_path = model_folder / "config.json"
    if not config_path.exists():
        return None

    config = load_json(config_path)
    pl.seed_everything(int(config.get("seed", 37)), workers=True)

    # Data
    if "data" not in config:
        raise KeyError(f"config.json missing 'data' field: {config_path}")
    data_cfg = dict(config["data"])

    dm = build_datamodule(dataset=dataset, task=task, data_cfg=data_cfg)
    dm.prepare_data()
    dm.setup()

    # Checkpoint
    ckpts = sorted([f for f in os.listdir(model_folder) if f.endswith(".ckpt")])
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .ckpt in {model_folder}")
    ckpt_path = model_folder / ckpts[0]

    # Load model
    if task == "survival":
        model = RNA_SurvivalModel.load_from_checkpoint(str(ckpt_path))
    else:
        model = RNA_ResponseModel.load_from_checkpoint(str(ckpt_path))

    model.eval()

    # Single-device trainer for prediction
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

    results = {}

    if task == "survival":
        _, risk, time, event, organ, cancer_type = _consolidate_survival(pred_batches)

        event_bool = np.asarray(event, dtype=bool)
        risk = np.asarray(risk, dtype=float)
        time = np.asarray(time, dtype=float)
        organ = list(organ)
        cancer_type = list(cancer_type)

        results["overall"] = safe_cindex(event_bool, time, risk)

        # per cancer type
        for ct in set(cancer_type):
            if ct is None:
                continue
            idx = [i for i, ci in enumerate(cancer_type) if ci == ct]
            if len(idx) < min_n:
                continue
            if int(np.sum(event_bool[idx])) < min_events:
                continue
            results[str(ct)] = safe_cindex(event_bool[idx], time[idx], risk[idx])

    else:
        _, score, y_true, organ = _consolidate_response(pred_batches)

        y_true = np.asarray(y_true, dtype=int)
        score = np.asarray(score, dtype=float)
        organ = list(organ)

        # Overall metrics
        results["overall_auroc"] = safe_auroc(y_true, score)
        results["overall_f1"] = safe_f1_macro(y_true, score, threshold=threshold)

        # Keep backwards compatibility: "overall" stays AUROC
        results["overall"] = results["overall_auroc"]

        # Per-organ metrics
        for o in set(organ):
            if o is None:
                continue
            idx = [i for i, oi in enumerate(organ) if oi == o]
            if len(idx) < min_n:
                continue

            results[f"{o}_auroc"] = safe_auroc(y_true[idx], score[idx])
            results[f"{o}_f1"] = safe_f1_macro(y_true[idx], score[idx], threshold=threshold)

    return results

def main(args):
    dataset = args.dataset
    task = args.task

    base_root = Path("/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs") / dataset

    if task == "survival":
        endpoint = args.endpoint
        base = base_root / "survival" / endpoint
        results_dir = Path(args.results_dir) if args.results_dir is not None else base
        out_csv = Path(args.out_csv) if args.out_csv is not None else (base / "all_metrics.csv")
        print(f"Dataset={dataset} task={task} endpoint={endpoint}")
    else:
        base = base_root / "response"
        results_dir = Path(args.results_dir) if args.results_dir is not None else base
        out_csv = Path(args.out_csv) if args.out_csv is not None else (base / "all_metrics.csv")
        print(f"Dataset={dataset} task={task} metrics=AUROC + macro-F1 (threshold={args.threshold})")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"Found {len(folders)} subfolders in {results_dir}")

    done_keys, existing_df = load_done_keys(out_csv)
    print(f"Loaded {len(done_keys)} existing entries from {out_csv}" if existing_df is not None else "No existing CSV found")

    new_rows = []
    skipped = 0

    for i, folder in tqdm(enumerate(folders, 1), total=len(folders)):
        print(f"\n[{i}/{len(folders)}] Evaluating: {folder.name}", flush=True)
        meta = parse_run_name(folder.name)
        if meta is None:
            continue

        # Duplicate detection key
        if existing_df is not None and "task" in existing_df.columns:
            key = (
                str(task),
                meta["arch"],
                meta["loss"],
                canon_rna_key(meta["rna_key"]),
                meta["n_hvgs"],
                int(meta["seed"]),
                int(meta["split"]),
            )
        else:
            key = (
                meta["arch"],
                meta["loss"],
                canon_rna_key(meta["rna_key"]),
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
                task=task,
                min_n=args.min_n,
                min_events=args.min_events,
                use_gpu=True,
                threshold=args.threshold
            )
        except Exception as e:
            row = {
                "run_dir": str(folder),
                "dataset": dataset,
                "task": task,
                "arch": meta["arch"],
                "loss": meta["loss"],
                "RNA_key": meta["rna_key"],
                "nHVGs": meta["n_hvgs"],
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
            "task": task,
            "arch": meta["arch"],
            "loss": meta["loss"],
            "RNA_key": meta["rna_key"],
            "nHVGs": meta["n_hvgs"],
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
            print(f"[{i}/{len(folders)}] {folder.name} overall={row['overall']}")

    print(f"Skipped {skipped} runs already in CSV")

    df_new = pd.DataFrame(new_rows)
    df_out = df_new if existing_df is None else pd.concat([existing_df, df_new], ignore_index=True, sort=False)

    # Column ordering: keep your old core fields, add dataset/task, then organs, then run_dir/error
    base_cols = ["dataset", "task", "arch", "loss", "RNA_key", "nHVGs", "seed", "split", "overall"]
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

    p.add_argument("--dataset", required=True, choices=["TCGA", "CPTAC", "CALGB", "Hartwig", "ICGC", "TARGET_AML"])
    p.add_argument("--task", required=True, choices=["survival", "response"])

    # survival-only (folder name); ignored for response
    p.add_argument("--endpoint", default="PFS", help="Only used for task=survival (folder name).")

    # optional overrides
    p.add_argument("--results_dir", default=None, help="Folder containing run subfolders")
    p.add_argument("--out_csv", default=None, help="Where to write the CSV")

    p.add_argument("--min_n", type=int, default=0)
    p.add_argument("--min_events", type=int, default=0, help="Only used for survival per-organ filtering.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for response macro-F1.")

    args = p.parse_args()
    main(args)
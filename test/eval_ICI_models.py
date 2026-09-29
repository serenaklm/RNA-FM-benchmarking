"""
test/eval_ICI_models.py

Evaluate all trained ICI response models under a results directory.

Metrics (matching COMPASS paper):
  AUROC     — area under ROC curve
  MCC       — Matthews Correlation Coefficient
  PR_AUC    — area under precision-recall curve
  Macro-F1  — for reference

Per-cohort slices are computed automatically from the 'cohort' field in each
sample's metadata (available when model was trained with ICI_merged as dataset).

Usage:
  # within-cohort (Hugo)
  python -m test.eval_ICI_models \\
    --paradigm within_cohort --cohort_tag within_Hugo

  # LOCO (held-out = Riaz)
  python -m test.eval_ICI_models \\
    --paradigm loco --cohort_tag loco_held_out_Riaz

  # Transfer (Hugo → Rose)
  python -m test.eval_ICI_models \\
    --paradigm transfer --cohort_tag transfer_Hugo_to_Rose

  # Override results_dir or out_csv explicitly
  python -m test.eval_ICI_models \\
    --paradigm loco --cohort_tag loco_held_out_Riaz \\
    --results_dir /path/to/runs --out_csv /path/to/metrics.csv

Assumes each run folder contains config.json + one .ckpt file (same as PFS eval).
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
)
from tqdm import tqdm

from pytorch_modules.utils.utils import load_json
from pytorch_modules.data.ici_data import ICIResponseDataModule
from pytorch_modules.models.survival_models import RNA_ResponseModel

# ── Same run-folder naming pattern as train_ICI_model.py ─────────────────────
#   {arch}_BCE_rnaKey_{rna_key}_nHVGs_{n}_seed_{seed}_split_{split}_{timestamp}
RUN_RE = re.compile(
    r"^(?P<arch>[^_]+)_BCE_rnaKey_(?P<rna_key>.+?)_nHVGs_(?P<n_hvgs>[\w]+)"
    r"_seed_(?P<seed>\d+)_split_(?P<split>\d+)_"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_OUTPUTS_ROOT = REPO_ROOT / "model_outputs"


# ── Metric helpers ────────────────────────────────────────────────────────────

def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _to_probs(scores):
    """Convert logits → probabilities if scores are outside [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    if np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0:
        return _sigmoid(scores)
    return scores


def safe_auroc(y_true, y_score):
    y_true  = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_prauc(y_true, y_score):
    y_true  = np.asarray(y_true, dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_mcc(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs  = _to_probs(y_score)
    y_pred = (probs >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float("nan")


def safe_f1_macro(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs  = _to_probs(y_score)
    y_pred = (probs >= threshold).astype(int)
    if len(y_true) < 2:
        return float("nan")
    try:
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float("nan")


def bootstrap_prauc_ci(y_true, y_score, n_boot=1000, ci=0.95, rng=None):
    """Bootstrap mean and CI for AUPRC. Returns (mean, lo, hi)."""
    y_true  = np.asarray(y_true,  dtype=int).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2 or len(y_true) < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng if rng is not None else 0)
    n   = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(average_precision_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return float(np.mean(vals)), float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))


def compute_metrics(y_true, y_score, threshold=0.5, n_boot=1000):
    pr_mean, pr_lo, pr_hi = bootstrap_prauc_ci(y_true, y_score, n_boot=n_boot)
    return {
        "auroc":      safe_auroc(y_true, y_score),
        "pr_auc":     safe_prauc(y_true, y_score),
        "pr_auc_lo":  pr_lo,
        "pr_auc_hi":  pr_hi,
        "mcc":        safe_mcc(y_true, y_score, threshold),
        "f1":         safe_f1_macro(y_true, y_score, threshold),
    }


# ── Parse run folder name ─────────────────────────────────────────────────────

def parse_run_name(folder_name: str):
    m = RUN_RE.search(folder_name)
    if not m:
        return None
    d = m.groupdict()
    d["n_hvgs"] = None if d["n_hvgs"] == "None" else int(d["n_hvgs"])
    d["seed"]   = int(d["seed"])
    d["split"]  = int(d["split"])
    return d


# ── Consolidate predictions ───────────────────────────────────────────────────

def _consolidate(pred_batches):
    """
    Collect (id_, score, y_true, cohort) from all prediction batches.
    predict_step returns (id_, logits, y, organ, cancer_type, cohort).
    """
    ids, scores, y_trues, cohorts = [], [], [], []
    for batch in pred_batches:
        id_, score, y_true, organ, cancer_type, cohort_val = batch
        n = len(score)
        ids.extend(list(id_) if id_ is not None else [None] * n)
        scores.extend(list(score))
        y_trues.extend(list(y_true))
        cohorts.extend(list(cohort_val) if cohort_val is not None else [None] * n)
    return ids, scores, y_trues, cohorts


# ── Evaluate one run folder ───────────────────────────────────────────────────

def eval_one_folder(model_folder: Path, use_gpu: bool, threshold: float, n_boot: int = 1000):
    config_path = model_folder / "config.json"
    if not config_path.exists():
        return None

    config = load_json(config_path)
    pl.seed_everything(int(config.get("seed", 37)), workers=True)

    data_cfg = dict(config["data"])

    # ICIResponseDataModule does not take 'endpoint'
    data_cfg.pop("endpoint", None)

    dm = ICIResponseDataModule(**data_cfg)
    dm.prepare_data()
    dm.setup()

    ckpts = sorted([f for f in os.listdir(model_folder) if f.endswith(".ckpt")])
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {model_folder}")
    ckpt_path = model_folder / ckpts[0]

    model = RNA_ResponseModel.load_from_checkpoint(str(ckpt_path))
    model.eval()

    trainer_kwargs = dict(config.get("trainer", {}))
    for k in ("devices", "accelerator", "strategy"):
        trainer_kwargs.pop(k, None)

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
        test_batches = trainer.predict(model, dataloaders=dm.test_dataloader())
        val_batches  = trainer.predict(model, dataloaders=dm.val_dataloader())

    def _eval_split(batches, prefix):
        _, scores, y_trues, cohorts = _consolidate(batches)
        scores  = np.asarray(scores,  dtype=float)
        y_trues = np.asarray(y_trues, dtype=int)
        out = {}
        m = compute_metrics(y_trues, scores, threshold, n_boot=n_boot)
        for k, v in m.items():
            out[f"{prefix}_{k}"] = v
        out[prefix] = out[f"{prefix}_pr_auc"]
        # per-cohort slices
        for c in [x for x in set(cohorts) if x is not None and x != "Unknown"]:
            idx = [i for i, ci in enumerate(cohorts) if ci == c]
            if len(idx) < 2:
                continue
            for k, v in compute_metrics(y_trues[idx], scores[idx], threshold, n_boot=n_boot).items():
                out[f"{c}_{prefix}_{k}"] = v
        return out

    results = {}
    results.update(_eval_split(test_batches, "overall"))
    results.update(_eval_split(val_batches,  "val"))
    return results


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def load_done_keys(out_csv: Path):
    if not out_csv.exists():
        return set(), None
    df = pd.read_csv(out_csv)
    needed = {"arch", "RNA_key", "nHVGs", "seed", "split"}
    if not needed.issubset(df.columns):
        return set(), df
    done = set(
        zip(
            df["arch"].astype(str),
            df["RNA_key"].astype(str),
            df["nHVGs"].apply(lambda x: None if pd.isna(x) else int(x)),
            df["seed"].astype(int),
            df["split"].astype(int),
        )
    )
    return done, df


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    base = MODEL_OUTPUTS_ROOT / "ICI" / args.paradigm / args.cohort_tag

    results_dir = Path(args.results_dir) if args.results_dir else base
    out_csv     = Path(args.out_csv)     if args.out_csv     else (base / "all_metrics.csv")

    print(f"Paradigm   : {args.paradigm}")
    print(f"Cohort tag : {args.cohort_tag}")
    print(f"Results dir: {results_dir}")
    print(f"Output CSV : {out_csv}")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"Found {len(folders)} run folders")

    done_keys, existing_df = load_done_keys(out_csv)
    if existing_df is not None:
        print(f"Loaded {len(done_keys)} existing entries")

    new_rows = []
    skipped  = 0

    for i, folder in tqdm(enumerate(folders, 1), total=len(folders)):
        meta = parse_run_name(folder.name)
        if meta is None:
            continue

        key = (
            meta["arch"],
            str(meta["rna_key"]),
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
            res = eval_one_folder(folder, use_gpu=True, threshold=args.threshold, n_boot=args.n_boot)
        except Exception as e:
            row = {
                "paradigm": args.paradigm,
                "cohort_tag": args.cohort_tag,
                "arch": meta["arch"],
                "RNA_key": meta["rna_key"],
                "nHVGs": meta["n_hvgs"],
                "seed": meta["seed"],
                "split": meta["split"],
                "overall": np.nan,
                "error": str(e),
                "run_dir": str(folder),
            }
            new_rows.append(row)
            continue

        if res is None:
            if args.verbose:
                print(f"SKIP (no config.json): {folder.name}")
            continue

        row = {
            "paradigm":   args.paradigm,
            "cohort_tag": args.cohort_tag,
            "arch":       meta["arch"],
            "RNA_key":    meta["rna_key"],
            "nHVGs":      meta["n_hvgs"],
            "seed":       meta["seed"],
            "split":      meta["split"],
            "overall":    res.get("overall", np.nan),
            "error":      "",
            "run_dir":    str(folder),
        }
        for k, v in res.items():
            if k != "overall":
                row[k] = v

        new_rows.append(row)
        if args.verbose:
            print(f"[{i}] {folder.name}  pr_auc={row.get('overall_pr_auc', np.nan):.3f}  "
                  f"auroc={row.get('overall_auroc', np.nan):.3f}  "
                  f"mcc={row.get('overall_mcc', np.nan):.3f}")

    print(f"Skipped {skipped} already-evaluated runs")

    df_new = pd.DataFrame(new_rows)
    df_out = (
        df_new if existing_df is None
        else pd.concat([existing_df, df_new], ignore_index=True, sort=False)
    )

    id_cols    = ["paradigm", "cohort_tag", "arch", "RNA_key", "nHVGs", "seed", "split"]
    base_cols  = id_cols + ["overall", "overall_auroc", "overall_pr_auc", "overall_pr_auc_lo", "overall_pr_auc_hi", "overall_mcc", "overall_f1"]
    extra_cols = ["run_dir", "error"]
    other_cols = sorted([c for c in df_out.columns if c not in set(base_cols + extra_cols)])
    cols = base_cols + other_cols + extra_cols

    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} rows → {out_csv}")

    # Write val_metrics.csv (used for arch selection in the analysis notebook)
    val_cols = [c for c in df_out.columns if c.startswith("val_")]
    if val_cols:
        val_df = df_out[id_cols + val_cols].copy()
        val_csv = out_csv.parent / "val_metrics.csv"
        val_df.to_csv(val_csv, index=False)
        print(f"Wrote val metrics → {val_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--paradigm", required=True,
        choices=["within_cohort", "loco", "transfer"],
    )
    p.add_argument(
        "--cohort_tag", required=True, type=str,
        help="Must match the cohort_tag used during training (e.g. 'within_Hugo', 'loco_held_out_Riaz').",
    )
    p.add_argument("--results_dir", default=None)
    p.add_argument("--out_csv",     default=None)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--n_boot",      type=int,   default=1000,
                   help="Bootstrap resamples for AUPRC CI (0 to disable).")
    p.add_argument("--verbose",     action="store_true")

    args = p.parse_args()
    main(args)

"""
test/eval_compass_ICI.py

Evaluate all COMPASS ICI runs under model_outputs/ICI/compass/{paradigm}/{tag}/.

Each run folder must contain:
  predictions.csv  — columns: sample_id, score, response, cohort, split
  config.json      — columns: paradigm, cohort_tag, mode, seed, split, ...

Metrics (matching COMPASS paper):
  AUROC, MCC, PR_AUC, Macro-F1

Usage:
  python -m test.eval_compass_ICI \\
      --paradigm loco --cohort_tag held_out_Riaz

  python -m test.eval_compass_ICI \\
      --paradigm within_cohort --cohort_tag Hugo

  python -m test.eval_compass_ICI \\
      --paradigm transfer --cohort_tag Hugo_to_Rose
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from tqdm import tqdm

OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "model_outputs" / "ICI" / "compass"


# ── Metric helpers ────────────────────────────────────────────────────────────

def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _to_probs(scores):
    scores = np.asarray(scores, dtype=float)
    if np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0:
        return _sigmoid(scores)
    return scores


def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_prauc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_mcc(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int).ravel()
    probs  = _to_probs(y_score)
    y_pred = (probs >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        return float("nan")


def safe_f1_macro(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int).ravel()
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
    y_true  = np.asarray(y_true,  dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
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


# ── Evaluate one run folder ───────────────────────────────────────────────────

def eval_one_folder(run_dir: Path, threshold: float, n_boot: int = 1000):
    pred_path   = run_dir / "predictions.csv"
    config_path = run_dir / "config.json"

    if not pred_path.exists() or not config_path.exists():
        return None

    preds = pd.read_csv(pred_path)
    with open(config_path) as f:
        config = json.load(f)

    results = {}

    for split_name, prefix in [("test", "overall"), ("val", "val")]:
        sub = preds[preds["split"] == split_name]
        if sub.empty:
            continue
        y_true  = sub["response"].values
        y_score = sub["score"].values

        m = compute_metrics(y_true, y_score, threshold, n_boot=n_boot)
        for k, v in m.items():
            results[f"{prefix}_{k}"] = v
        results[prefix] = results[f"{prefix}_pr_auc"]

        # Per-cohort slices
        if "cohort" in sub.columns:
            for cohort in sub["cohort"].dropna().unique():
                if cohort == "Unknown":
                    continue
                idx = sub["cohort"] == cohort
                if idx.sum() < 2:
                    continue
                for k, v in compute_metrics(
                    y_true[idx.values], y_score[idx.values], threshold, n_boot=n_boot
                ).items():
                    results[f"{cohort}_{prefix}_{k}"] = v

    return results, config


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def load_done_keys(out_csv: Path):
    if not out_csv.exists():
        return set(), None
    df = pd.read_csv(out_csv)
    needed = {"mode", "seed", "split"}
    if not needed.issubset(df.columns):
        return set(), df
    done = set(zip(df["mode"].astype(str), df["seed"].astype(int), df["split"].astype(int)))
    return done, df


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    base        = OUTPUTS_ROOT / args.paradigm / args.cohort_tag
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

    for folder in tqdm(folders):
        ret = eval_one_folder(folder, args.threshold, n_boot=args.n_boot)
        if ret is None:
            continue

        results, config = ret

        key = (config.get("mode", ""), int(config.get("seed", 0)), int(config.get("split", 0)))
        if key in done_keys:
            skipped += 1
            continue

        row = {
            "paradigm":   args.paradigm,
            "cohort_tag": args.cohort_tag,
            "mode":       config.get("mode", ""),
            "seed":       config.get("seed", np.nan),
            "split":      config.get("split", np.nan),
            "overall":    results.get("overall", np.nan),
            "error":      "",
            "run_dir":    str(folder),
        }
        for k, v in results.items():
            if k != "overall":
                row[k] = v

        new_rows.append(row)
        if args.verbose:
            print(f"  {folder.name}  pr_auc={row.get('overall_pr_auc', np.nan):.3f}  "
                  f"auroc={row.get('overall_auroc', np.nan):.3f}  "
                  f"mcc={row.get('overall_mcc', np.nan):.3f}")

    print(f"Skipped {skipped} already-evaluated runs")

    df_new = pd.DataFrame(new_rows)
    df_out = (
        df_new if existing_df is None
        else pd.concat([existing_df, df_new], ignore_index=True, sort=False)
    )

    id_cols   = ["paradigm", "cohort_tag", "mode", "seed", "split"]
    base_cols = id_cols + ["overall", "overall_auroc", "overall_pr_auc", "overall_pr_auc_lo", "overall_pr_auc_hi", "overall_mcc", "overall_f1"]
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

    # val_metrics.csv for model selection
    val_cols = [c for c in df_out.columns if c.startswith("val_")]
    if val_cols:
        val_df  = df_out[id_cols + val_cols].copy()
        val_csv = out_csv.parent / "val_metrics.csv"
        val_df.to_csv(val_csv, index=False)
        print(f"Wrote val metrics → {val_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm",    required=True,
                   choices=["within_cohort", "loco", "transfer"])
    p.add_argument("--cohort_tag",  required=True, type=str)
    p.add_argument("--results_dir", default=None)
    p.add_argument("--out_csv",     default=None)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--n_boot",      type=int,   default=1000,
                   help="Bootstrap resamples for AUPRC CI (0 to disable).")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()
    main(args)

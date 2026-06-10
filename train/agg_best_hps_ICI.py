"""
train/agg_best_hps_ICI.py

Aggregate per-split best hyperparameters from ICI Optuna tuning into a single
JSON file consumed by train_ICI_model.py via --hp_path.

HP key format:   {rna_embedding_key}_{arch}_BCE
  e.g.  BulkFormer_ResidualMLP_BCE
        RNA_seq_500_Linear_BCE
        ssGSEA_immuno_Linear_BCE

Aggregation:
  For each (rna_embedding_key, arch), take the best trial per split
  (min val_loss), then compute:
    - geometric mean across splits for log-scale HPs (lr, weight_decay, penalty_weight)
    - arithmetic mean across splits for linear-scale HPs (dropout_rate)

Usage:
    python -m train.agg_best_hps_ICI

Output:
    hp_results_ICI/all_aggregated_hps_ICI.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HP_DIR   = REPO_ROOT / "hp_results_ICI"
CSV_PATH = HP_DIR / "tuning_history_ICI.csv"
OUT_PATH = HP_DIR / "all_aggregated_hps_ICI.json"

# Same 11 embeddings as PFS benchmarking (matches tune_ICI_models.sh)
EXPECTED_EMBEDDINGS = [
    "BulkFormer",
    "BulkRNABert",
    "COMPASS",
    "scGPT",
    "STATE",
    "CancerFoundation",
    "RNA_seq_1250",     # HVG representative; HPs shared across all HVG counts
    "ssGSEA_combined",  # ssGSEA pooling representative; HPs shared across all ssGSEA_* keys
    "max_combined",     # max pooling representative; HPs shared across all max_* keys
    "PCA_50",           # PCA representative; HPs shared with PCA_100
    "RNA_seq",          # raw expression (all genes)
]
ARCHS    = ["Linear", "ResidualMLP"]
SPLITS   = [0, 1, 2, 3, 4]
N_TRIALS = 50          # expected minimum trials per run


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run tune_ICI_models.sh first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} trial rows from {CSV_PATH}")

    # ── Completion check ──────────────────────────────────────────────────────
    completed = (
        df.groupby(["rna_embedding_key", "arch", "split_idx"])
        .size()
        .reset_index(name="n_trials")
    )
    expected = pd.MultiIndex.from_product(
        [EXPECTED_EMBEDDINGS, ARCHS, SPLITS],
        names=["rna_embedding_key", "arch", "split_idx"],
    ).to_frame(index=False)

    merged = expected.merge(
        completed, on=["rna_embedding_key", "arch", "split_idx"], how="left"
    )
    merged["n_trials"] = merged["n_trials"].fillna(0).astype(int)
    merged["complete"] = merged["n_trials"] >= N_TRIALS

    n_complete = int(merged["complete"].sum())
    n_total    = len(merged)
    print(f"\nCompletion: {n_complete} / {n_total} runs have ≥{N_TRIALS} trials")

    incomplete = merged[~merged["complete"]].sort_values(
        ["rna_embedding_key", "arch", "split_idx"]
    )
    if not incomplete.empty:
        print(f"\nIncomplete runs ({len(incomplete)}):")
        print(incomplete[["rna_embedding_key", "arch", "split_idx", "n_trials"]].to_string(index=False))
        print("\nWARNING: aggregating from partial results — rerun tune_ICI_models.sh for missing entries.\n")

    # ── Best trial per (key, arch, split) ─────────────────────────────────────
    df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
    best_per_split = df.loc[
        df.groupby(["rna_embedding_key", "arch", "split_idx"])["val_loss"].idxmin()
    ].copy()

    # ── Aggregate across splits ───────────────────────────────────────────────
    LOG_HPS    = ["lr", "weight_decay", "penalty_weight"]
    LINEAR_HPS = ["dropout_rate"]

    for hp in LOG_HPS + LINEAR_HPS:
        best_per_split[hp] = pd.to_numeric(
            best_per_split[hp].replace("", float("nan")), errors="coerce"
        )

    def agg_group(group):
        result = {}
        for hp in LOG_HPS:
            vals = group[hp].dropna().values.astype(float)
            result[hp] = float(np.exp(np.mean(np.log(vals)))) if len(vals) > 0 else float("nan")
        for hp in LINEAR_HPS:
            vals = group[hp].dropna().values.astype(float)
            result[hp] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
        return pd.Series(result)

    aggregated = (
        best_per_split.groupby(["rna_embedding_key", "arch"])
        .apply(agg_group)
        .reset_index()
    )

    # ── Build output dict ─────────────────────────────────────────────────────
    all_hps = {}
    for _, row in aggregated.iterrows():
        key = f"{row['rna_embedding_key']}_{row['arch']}_BCE"
        hps = {
            "lr":             row["lr"],
            "weight_decay":   row["weight_decay"],
            "penalty_weight": row["penalty_weight"],
        }
        if row["arch"] != "Linear":
            hps["dropout_rate"] = row["dropout_rate"]
        all_hps[key] = hps

    # ── Save ──────────────────────────────────────────────────────────────────
    HP_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_hps, f, indent=2)

    print(f"\nAggregated HPs saved → {OUT_PATH}")
    print(f"  {len(all_hps)} keys written:\n")
    for key, hps in sorted(all_hps.items()):
        hp_str = "  ".join(f"{k}={v:.2e}" for k, v in hps.items())
        print(f"    {key:<45}  {hp_str}")

    print(f"""
To use tuned HPs in training, pass --hp_path to the training scripts:

  HP_PATH=hp_results_ICI/all_aggregated_hps_ICI.json \\
    ./scripts/run_within_4gpu.sh

Or set hp_path in configs/ici_model_config.yaml.
""")


if __name__ == "__main__":
    main()

import pandas as pd
import os
import glob
import numpy as np
import json

CSV_PATH = "/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/tuning_history.csv"
LOG_DIR = "/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results"

RNA_KEYS = ["BulkFormer", "BulkRNABert", "COMPASS", "scGPT", "STATE", "CancerFoundation", "RNA_seq_1250", "RNA_seq_17490", "ssGSEA_combined", "max_combined", "PCA_50"]
ARCHS = ["ResidualMLP", "Linear"]
LOSSES = ["Cox", "CIndex"]
SPLITS = [0, 1, 2, 3, 4]
N_TRIALS = 50

# Load CSV
df = pd.read_csv(CSV_PATH)

# Count completed trials per (key, arch, loss, split)
completed = df.groupby(["rna_embedding_key", "arch", "loss_name", "split_idx"]).size().reset_index(name="n_trials")

# Build full expected set
expected = pd.MultiIndex.from_product(
    [RNA_KEYS, ARCHS, LOSSES, SPLITS],
    names=["rna_embedding_key", "arch", "loss_name", "split_idx"]
).to_frame(index=False)

# Merge to find missing/incomplete
merged = expected.merge(completed, on=["rna_embedding_key", "arch", "loss_name", "split_idx"], how="left")
merged["n_trials"] = merged["n_trials"].fillna(0).astype(int)
merged["complete"] = merged["n_trials"] >= N_TRIALS
merged["over"] = merged["n_trials"] > N_TRIALS

print(f"Complete: {merged['complete'].sum()} / {len(merged)}")
print(f"Over-complete: {merged['over'].sum()} / {len(merged)}\n")
print(f"Missing/incomplete:\n")

incomplete = merged[~merged["complete"]].sort_values(["rna_embedding_key", "arch", "loss_name", "split_idx"])
print(incomplete.to_string(index=False))

# For each over-complete group, keep only the first 50 trials
def trim_to_n_trials(group, n=50):
    return group.head(n)

df_trimmed = df.groupby(
    ["rna_embedding_key", "arch", "loss_name", "split_idx"],
    group_keys=False
).apply(trim_to_n_trials)

print(f"Trimmed DataFrame has {len(df_trimmed)} from original {len(df)}")

# For each (key, arch, loss, split), get the best trial (min val_loss)
best_per_split = df.loc[df.groupby(["rna_embedding_key", "arch", "loss_name", "split_idx"])["val_loss"].idxmin()]

# Aggregate best HPs across splits
log_hps = ["lr", "weight_decay", "penalty_weight"]
linear_hps = ["dropout_rate"]

def aggregate_group(group):
    result = {}
    for hp in log_hps:
        vals = group[hp].replace("", np.nan).dropna().astype(float)
        result[hp] = float(np.exp(np.mean(np.log(vals)))) if len(vals) > 0 else np.nan
    for hp in linear_hps:
        vals = group[hp].replace("", np.nan).dropna().astype(float)
        result[hp] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return pd.Series(result)

aggregated_hps = best_per_split.groupby(["rna_embedding_key", "arch", "loss_name"]).apply(aggregate_group).reset_index()

all_hps = {}

for _, row in aggregated_hps.iterrows():
    key = f"{row['rna_embedding_key']}_{row['arch']}_{row['loss_name']}"
    all_hps[key] = {
        "lr": row["lr"],
        "weight_decay": row["weight_decay"],
        "penalty_weight": row["penalty_weight"],
    }
    if row["arch"] != "Linear":
        all_hps[key]["dropout_rate"] = row["dropout_rate"]

out_path = "/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/all_aggregated_hps.json"
with open(out_path, "w") as f:
    json.dump(all_hps, f, indent=2)

print(f"Saved to {out_path}")
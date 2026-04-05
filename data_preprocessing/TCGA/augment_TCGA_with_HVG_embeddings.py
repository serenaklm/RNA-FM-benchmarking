"""
Augments TCGA_processed.pkl with HVG-pooled foundation model embeddings.

New keys added to each sample dict:
  BulkRNABert_HVG_1250, BulkRNABert_HVG_5000, BulkRNABert_HVG_10000
  BulkFormer_HVG_1250,  BulkFormer_HVG_2000,  BulkFormer_HVG_5000, BulkFormer_HVG_10000

Patients with no embedding file get None for that key.
Output: data/RNA_embeddings/TCGA_processed_HVG.pkl
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EMB_ROOT  = REPO_ROOT / "data" / "TCGA" / "RNA" / "FM_embeddings"
IN_PKL    = REPO_ROOT / "data" / "RNA_embeddings" / "TCGA_processed.pkl"
OUT_PKL   = REPO_ROOT / "data" / "RNA_embeddings" / "TCGA_processed_HVG.pkl"

HVG_CONFIGS = {
    "BulkRNABert_HVG": [1250, 5000, 10000],
    "BulkFormer_HVG":  [1250, 2000, 5000, 10000],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_emb(path: Path) -> np.ndarray:
    emb = load_pickle(path)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"Loading {IN_PKL} ...")
data = load_pickle(IN_PKL)
patient_ids = list(data.keys())
print(f"Loaded {len(patient_ids)} patients.")

# Pre-index all available embedding files per (model, k) for fast lookup
emb_index = {}   # (model_key, k) -> {patient_id: Path}

for folder_name, sizes in HVG_CONFIGS.items():
    for k in sizes:
        subfolder = EMB_ROOT / folder_name / f"{k}_HVGs"
        if not subfolder.exists():
            print(f"  WARNING: {subfolder} not found, skipping.")
            emb_index[(folder_name, k)] = {}
            continue
        index = {}
        for pkl_file in subfolder.glob("*.pkl"):
            index[pkl_file.stem] = pkl_file   # stem == patient_id
        emb_index[(folder_name, k)] = index
        # Print dimensionality from the first available file
        sample_dim = None
        if index:
            sample_dim = load_emb(next(iter(index.values()))).shape
        print(f"  Indexed {len(index):5d} files  ->  {folder_name}/{k}_HVGs  (shape: {sample_dim})")

# Add HVG keys to each patient record
missing_counts = {(f, k): 0 for f, sizes in HVG_CONFIGS.items() for k in sizes}

for patient_id in tqdm(patient_ids, desc="Augmenting patients"):
    for folder_name, sizes in HVG_CONFIGS.items():
        # e.g. "BulkRNABert_HVG" -> key prefix "BulkRNABert_HVG"
        for k in sizes:
            dict_key = f"{folder_name}_{k}"   # e.g. "BulkRNABert_HVG_1250"
            index = emb_index[(folder_name, k)]
            if patient_id in index:
                data[patient_id][dict_key] = load_emb(index[patient_id])
            else:
                data[patient_id][dict_key] = None
                missing_counts[(folder_name, k)] += 1

# Summary
print("\nMissing embedding counts (patient had no file):")
for (folder_name, k), count in missing_counts.items():
    print(f"  {folder_name}_{k}: {count} / {len(patient_ids)} missing")

print(f"\nSaving augmented data to {OUT_PKL} ...")
pickle_data(data, OUT_PKL)
print("Done.")

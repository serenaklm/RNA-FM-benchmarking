"""
Extract and average weights from trained Linear CIndex survival models.

For each dataset, produces a dict with two keys (best HVG config and raw expression),
where values are dicts of {gene_name: avg_weight} sorted by descending |weight|.

Output: analysis/linear_gene_weights.pkl
"""

import os
import sys
import pickle
import glob
import numpy as np

# Add repo root to path for model imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from pytorch_modules.models.survival_models import RNA_SurvivalModel

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "TCGA": {
        "model_dir": os.path.join(REPO_ROOT, "model_outputs/TCGA/survival/PFS"),
        "gene_file": os.path.join(REPO_ROOT, "data/TCGA/TCGA_genes_ordered_by_variance.pkl"),
        "best_hvg": 10000,
    },
    "Hartwig": {
        "model_dir": os.path.join(REPO_ROOT, "model_outputs/Hartwig/survival/PFS"),
        "gene_file": os.path.join(REPO_ROOT, "data/Hartwig/Hartwig_genes_ordered_by_variance.pkl"),
        "best_hvg": 5000,
    },
    "CPTAC": {
        "model_dir": os.path.join(REPO_ROOT, "model_outputs/CPTAC/survival/PFS"),
        "gene_file": os.path.join(REPO_ROOT, "data/CPTAC/CPTAC_genes_ordered_by_variance.pkl"),
        "best_hvg": 1250,
    },
    "TARGET_AML": {
        "model_dir": os.path.join(REPO_ROOT, "model_outputs/TARGET_AML/survival/PFS"),
        "gene_file": os.path.join(REPO_ROOT, "data/TARGET_AML/TARGET_AML_genes_ordered_by_variance.pkl"),
        "best_hvg": 1250,
    },
}

SEEDS  = [37, 42]
SPLITS = [0, 1, 2, 3, 4]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_gene_names(gene_file: str) -> list[str]:
    """Return list of gene names ordered by variance (descending)."""
    with open(gene_file, "rb") as f:
        g = pickle.load(f)
    return list(g.keys()) if isinstance(g, dict) else list(g)


def find_checkpoint(folder: str) -> str | None:
    """Return path to the .ckpt file inside a model output folder."""
    ckpts = glob.glob(os.path.join(folder, "*.ckpt"))
    if not ckpts:
        return None
    # Prefer the one with lowest val_loss if multiple (shouldn't happen)
    return ckpts[0]


def find_model_folders(model_dir: str, n_hvgs: int | None, seeds: list, splits: list) -> list[str]:
    """
    Find all model output folders matching:
        Linear_CIndex_rnaKey_RNA_seq_nHVGs_{n_hvgs}_seed_{seed}_split_{split}_*
    """
    hvg_str = str(n_hvgs) if n_hvgs is not None else "None"
    pattern = f"Linear_CIndex_rnaKey_RNA_seq_nHVGs_{hvg_str}_seed_*_split_*"
    all_folders = glob.glob(os.path.join(model_dir, pattern))

    matched = []
    for folder in all_folders:
        name = os.path.basename(folder)
        for seed in seeds:
            for split in splits:
                tag = f"_seed_{seed}_split_{split}_"
                if tag in name:
                    matched.append(folder)
    return matched


def extract_weights(folder: str) -> np.ndarray | None:
    """Load checkpoint and return weight vector as numpy array (rna_dim,)."""
    ckpt_path = find_checkpoint(folder)
    if ckpt_path is None:
        print(f"  [WARN] No checkpoint in {folder}")
        return None
    try:
        model = RNA_SurvivalModel.load_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        w = model.model.weight.detach().squeeze().numpy()  # (rna_dim,)
        return w
    except Exception as e:
        print(f"  [WARN] Failed to load {ckpt_path}: {e}")
        return None


def weights_to_gene_dict(avg_weights: np.ndarray, gene_names: list[str]) -> dict[str, float]:
    """
    Map averaged weight vector to gene names and sort by descending |weight|.
    avg_weights has length == len(gene_names) or n_HVGs (a prefix of gene_names).
    """
    n = len(avg_weights)
    assert n <= len(gene_names), (
        f"Weight vector length {n} exceeds gene list length {len(gene_names)}"
    )
    gene_subset = gene_names[:n]
    paired = sorted(zip(gene_subset, avg_weights.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return dict(paired)


# ── Main ───────────────────────────────────────────────────────────────────────

def process_dataset(dataset: str, cfg: dict) -> dict:
    gene_names = load_gene_names(cfg["gene_file"])
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  ({len(gene_names)} genes total)")

    result = {}

    for label, n_hvgs in [("raw_expression", None), (f"HVG_{cfg['best_hvg']}", cfg["best_hvg"])]:
        hvg_str = str(n_hvgs) if n_hvgs is not None else "None"
        print(f"\n  Config: nHVGs={hvg_str} → '{label}'")

        folders = find_model_folders(cfg["model_dir"], n_hvgs, SEEDS, SPLITS)
        print(f"  Found {len(folders)} matching folders (expected {len(SEEDS)*len(SPLITS)})")

        weight_list = []
        for folder in sorted(folders):
            w = extract_weights(folder)
            if w is not None:
                expected_dim = n_hvgs if n_hvgs is not None else len(gene_names)
                if len(w) != expected_dim:
                    print(f"  [WARN] Unexpected weight dim {len(w)} (expected {expected_dim}) in {os.path.basename(folder)}")
                    continue
                weight_list.append(w)

        if not weight_list:
            print(f"  [ERROR] No valid weights extracted for {label}")
            continue

        avg_w = np.mean(np.stack(weight_list, axis=0), axis=0)  # (rna_dim,)
        print(f"  Averaged {len(weight_list)} weight vectors, shape={avg_w.shape}")

        result[label] = weights_to_gene_dict(avg_w, gene_names)

    return result


def main():
    out_path = os.path.join(REPO_ROOT, "analysis", "linear_gene_weights.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_results = {}
    for dataset, cfg in DATASET_CONFIG.items():
        all_results[dataset] = process_dataset(dataset, cfg)

    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"\n{'='*60}")
    print(f"Saved to {out_path}")

    # Quick sanity check
    print("\nSummary:")
    for ds, res in all_results.items():
        for key, gene_dict in res.items():
            top5 = list(gene_dict.items())[:5]
            print(f"  {ds}/{key}: {len(gene_dict)} genes, top-5: {top5}")


if __name__ == "__main__":
    main()

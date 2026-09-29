"""
Pre-tokenize RNA-seq data for STATE-SE fine-tuning.

Mirrors the VCIDatasetSentenceCollator.sample_cell_sentences logic from the
STATE repo, but deterministically (no random subsampling) so tokens can be
pre-computed once and reused across training runs.

For each patient we produce a "cell sentence":
  - position 0: placeholder (overridden by learnable cls_token in forward)
  - positions 1..: top-(pad_length-1) genes sorted by expression (descending)
  - cell_sentence_counts: 100 × expression-weight for each position

If a patient has < pad_length-1 expressed HVG genes (rare for bulk RNA-seq),
remaining slots are filled with randomly sampled unexpressed genes from the
full protein-embedding gene universe.

Run from RNA-FM-benchmarking root with the state conda env:
  cd data_preprocessing/tokenize_for_finetuning/
  python STATE_tokenize.py --data_set TCGA
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import anndata
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_REPO_DIR = os.environ.get("STATE_REPO_DIR", "/data/rbg/users/ashyamal/state/src")
STATE_MODEL_DIR = os.environ.get("STATE_MODEL_DIR", str(REPO_ROOT / "data" / "STATE" / "SE-600M"))

PAD_LENGTH = 2048   # cfg.dataset.pad_length


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_protein_embeds(pe_path):
    """Load protein_embeddings.pt and return (gene_list, gene_to_global_idx)."""
    import torch
    pe = torch.load(pe_path, map_location="cpu", weights_only=False)
    gene_list = list(pe.keys())
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    return gene_list, gene_to_idx


def tokenize_patient(expr_log1p, gene_global_idxs, pad_length, rng, all_gene_idxs):
    """
    Build a single STATE cell sentence from log1p expression.

    Parameters
    ----------
    expr_log1p : np.ndarray, shape (n_genes,)
        Log1p-transformed expression for the patient's HVG genes.
    gene_global_idxs : np.ndarray, shape (n_genes,), dtype int64
        Global protein-embedding index for each HVG gene.
    pad_length : int
        Total cell-sentence length (including position 0 / CLS placeholder).
    rng : np.random.Generator
        For reproducible random padding when n_genes < pad_length - 1.
    all_gene_idxs : np.ndarray
        All valid global gene indices (for random padding of unexpressed slots).

    Returns
    -------
    cell_sentence : np.ndarray, shape (pad_length,), dtype int64
    cell_sentence_counts : np.ndarray, shape (pad_length,), dtype float32
    """
    n_genes = len(expr_log1p)

    # Undo log1p to get a measure proportional to counts
    expr_raw = np.expm1(expr_log1p.astype(np.float64))
    total = expr_raw.sum()
    if total > 0:
        expr_weights = (expr_raw / total).astype(np.float32)
    else:
        expr_weights = np.full(n_genes, 1.0 / n_genes, dtype=np.float32)

    # Sort genes by expression (descending)
    order = np.argsort(expr_log1p)[::-1]   # largest expression first

    cell_sentence = np.zeros(pad_length, dtype=np.int64)
    cell_sentence_counts = np.zeros(pad_length, dtype=np.float32)

    # Position 0: CLS placeholder (will be overridden in forward pass)
    # Leave as 0; the forward pass replaces it with the learnable cls_token.
    cell_sentence_counts[0] = 0.0

    available_slots = pad_length - 1   # positions 1 .. pad_length-1

    if n_genes >= available_slots:
        # Take the top available_slots genes
        top_order = order[:available_slots]
        cell_sentence[1:] = gene_global_idxs[top_order]
        cell_sentence_counts[1:] = 100.0 * expr_weights[top_order]
    else:
        # Fill with all HVG genes
        cell_sentence[1 : n_genes + 1] = gene_global_idxs[order]
        cell_sentence_counts[1 : n_genes + 1] = 100.0 * expr_weights[order]

        # Pad remaining slots with randomly sampled genes from the full universe
        remaining = available_slots - n_genes
        extra_idxs = rng.choice(all_gene_idxs, size=remaining, replace=True)
        cell_sentence[n_genes + 1 :] = extra_idxs
        # expression weight for padded positions is 0
        cell_sentence_counts[n_genes + 1 :] = 0.0

    return cell_sentence, cell_sentence_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        choices=["TCGA", "CPTAC", "CALGB", "Hartwig", "ICGC", "TARGET_AML"],
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible padding of unexpressed gene slots."
    )
    args = parser.parse_args()
    dataset = args.data_set
    rng = np.random.default_rng(args.seed)

    # ---- paths ----
    # HVG h5ad produced by STATE.py --action process_hvg
    key = f"{dataset}_bulk_RNA_processed_log1p"
    hvg_path = f"../../data/{dataset}/RNA/{key}_HVG.h5ad"
    pe_path = os.path.join(STATE_MODEL_DIR, "protein_embeddings.pt")
    output_dir = "../../data/tokens/STATE"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_tokens.pkl")

    # ---- load protein embeddings vocabulary ----
    print(f"Loading protein embeddings from {pe_path} ...")
    gene_list, gene_to_idx = load_protein_embeds(pe_path)
    all_gene_idxs = np.arange(len(gene_list), dtype=np.int64)
    print(f"Protein embedding vocabulary: {len(gene_list)} genes")

    # ---- load HVG expression data ----
    print(f"Reading HVG h5ad from {hvg_path} ...")
    adata = anndata.read_h5ad(hvg_path)
    print(f"Shape: {adata.shape}  ({adata.n_obs} patients × {adata.n_vars} HVG genes)")

    # ---- patient IDs (same logic as STATE.py) ----
    if dataset == "TARGET_AML":
        sample_ids = adata.obs_names.astype(str).tolist()
    else:
        obs_key = "cases.submitter_id" if dataset == "TCGA" else "patient_id"
        sample_ids = adata.obs[obs_key].astype(str).tolist()

    # ---- map HVG gene names → global indices ----
    hvg_gene_names = adata.var_names.tolist()
    hvg_global_idxs = np.array([gene_to_idx.get(g, -1) for g in hvg_gene_names], dtype=np.int64)

    n_matched = (hvg_global_idxs >= 0).sum()
    print(f"HVG genes matched to protein embeddings: {n_matched} / {len(hvg_gene_names)}")
    if n_matched == 0:
        raise RuntimeError(
            "No HVG genes matched protein embeddings.  "
            "Did you run STATE.py --action process_hvg first?"
        )

    # Only keep genes that have a valid global index
    valid_mask = hvg_global_idxs >= 0
    valid_idxs = np.where(valid_mask)[0]

    # ---- dense expression matrix ----
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # ---- tokenize each patient ----
    tokens_dict = {}
    for i, pid in tqdm(enumerate(sample_ids), total=len(sample_ids), desc="Tokenizing"):
        expr_full = X[i]
        expr_hvg = expr_full[valid_mask]
        gene_global = hvg_global_idxs[valid_mask]

        cell_sentence, cell_sentence_counts = tokenize_patient(
            expr_log1p=expr_hvg,
            gene_global_idxs=gene_global,
            pad_length=PAD_LENGTH,
            rng=rng,
            all_gene_idxs=all_gene_idxs,
        )
        tokens_dict[pid] = {
            "cell_sentence": cell_sentence,           # (2048,) int64
            "cell_sentence_counts": cell_sentence_counts,  # (2048,) float32
        }

    save_pickle(tokens_dict, output_path)
    print(f"Saved {len(tokens_dict)} samples → {output_path}")
    print(f"Cell sentence shape: {cell_sentence.shape}, counts shape: {cell_sentence_counts.shape}")

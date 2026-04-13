"""
Pre-tokenize RNA-seq data for CancerFoundation fine-tuning.

Mirrors CancerFoundation/model/embedding.py preprocessing exactly:
  1. Map gene symbols to CF vocab IDs; keep only in-vocab genes.
  2. Select top 1199 highly variable genes (HVGs) using scanpy
     (batch_key="project.project_id" for TCGA; dummy batch for others).
  3. For each patient: store the shared HVG gene IDs + patient-specific expressions.
     Prepend <cls> token (expression gets pad_value=-2 at cls position).
  4. Raw log1p expression values are stored; binning is deferred to the collate_fn
     so it matches CancerFoundation inference exactly.

Saves a single pickle:
    {patient_id: {"gene_ids": np.int64 array, "expressions": np.float32 array}}

All patients in a dataset share the same gene_ids (the HVG vocab IDs);
only expressions differ.

Run from: data_preprocessing/tokenize_for_finetuning/
Example (cancerfoundation conda env):
  python CancerFoundation_tokenize.py --data_set TCGA
  python CancerFoundation_tokenize.py --data_set CPTAC
"""

import os
import sys
import pickle
import argparse
import yaml
from pathlib import Path

import anndata
import numpy as np
import scanpy as sc
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
_data_paths = yaml.safe_load(open(REPO_ROOT / "configs" / "data_paths.yaml"))
CF_DIR = _data_paths["model_repos"]["CancerFoundation"]
sys.path.insert(0, CF_DIR)

from model.vocab import GeneVocab


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        choices=["TCGA", "CPTAC", "CALGB", "Hartwig", "ICGC", "TARGET_AML"],
    )
    args = parser.parse_args()
    dataset = args.data_set

    RNA_counts_path = f"../../data/{dataset}/RNA/{dataset}_bulk_RNA_processed_log1p.h5ad"
    vocab_file = os.path.join(CF_DIR, "model/assets/vocab.json")
    output_dir = "../../data/tokens/CancerFoundation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_tokens.pkl")

    # ---- load vocab ----
    vocab = GeneVocab.from_file(vocab_file)
    vocab.set_default_index(vocab["<pad>"])
    cls_id = vocab["<cls>"]
    pad_value = -2  # matches embedding.py

    print(f"CancerFoundation vocab size: {len(vocab)}")

    # ---- load RNA data ----
    adata = anndata.read_h5ad(RNA_counts_path)
    print(f"Loaded adata: {adata.shape}")

    # ---- filter to in-vocab genes (mirrors embedding.py) ----
    adata.var["genes"] = adata.var.index
    adata.var["id_in_vocab"] = [
        vocab[g] if g in vocab else -1 for g in adata.var["genes"]
    ]
    n_total = adata.n_vars
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    print(f"Genes in CF vocab: {adata.n_vars} / {n_total}")

    # ---- HVG selection (mirrors embedding.py exactly) ----
    max_length = 1200
    if dataset == "TCGA":
        batch_key = "project.project_id"
    else:
        adata.obs["batch_dummy"] = f"{dataset}_Study"
        batch_key = "batch_dummy"

    sc.pp.highly_variable_genes(
        adata, n_top_genes=max_length - 1, flavor="cell_ranger", batch_key=batch_key
    )
    adata = adata[:, adata.var["highly_variable"]]
    print(f"HVGs selected: {adata.n_vars}")

    # ---- vocab IDs for HVGs (shared by all patients) ----
    hvg_gene_names = adata.var.index.tolist()
    hvg_gene_ids = np.array([vocab[g] for g in hvg_gene_names], dtype=np.int64)

    # prepend <cls>
    gene_ids_with_cls = np.concatenate([[cls_id], hvg_gene_ids]).astype(np.int64)
    print(f"Sequence length (incl. <cls>): {len(gene_ids_with_cls)}")

    count_matrix = adata.X
    if hasattr(count_matrix, "toarray"):
        count_matrix = count_matrix.toarray()
    count_matrix = count_matrix.astype(np.float32)  # (n_patients, n_hvgs)

    # ---- patient IDs ----
    if dataset == "TARGET_AML":
        ids = adata.obs_names.astype(str).tolist()
    else:
        obs = "cases.submitter_id" if dataset == "TCGA" else "patient_id"
        ids = adata.obs[obs].astype(str).tolist()

    print(f"Tokenizing {len(ids)} patients for {dataset}")

    # ---- build token dict ----
    # gene_ids are identical for all patients; we store a reference (not a copy)
    # so pickle deduplicates the array via its memo mechanism.
    tokens_dict = {}
    for i, pid in tqdm(enumerate(ids), total=len(ids)):
        row = count_matrix[i]  # (n_hvgs,)
        expr_with_cls = np.concatenate([[pad_value], row]).astype(np.float32)
        tokens_dict[pid] = {
            "gene_ids": gene_ids_with_cls,   # shared array — same HVGs for every patient
            "expressions": expr_with_cls,
        }

    save_pickle(tokens_dict, output_path)
    print(f"Saved {len(tokens_dict)} tokenized samples to {output_path}")
    print(f"Sequence length: {len(gene_ids_with_cls)}  (HVGs: {adata.n_vars})")

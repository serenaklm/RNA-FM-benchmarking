#!/usr/bin/env python3
import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import anndata as ad
from sklearn.decomposition import PCA


def load_gene_list(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return list(obj.keys()) if isinstance(obj, dict) else list(obj)


def save_per_sample_embeddings(Z, ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for vec, id_ in zip(Z, ids):
        with open(out_dir / f"{id_}.pkl", "wb") as f:
            pickle.dump(vec.astype(np.float32), f)


def main(args):

    DATA_H5AD_PATH = f"../data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"
    GENE_NAME_PATH = f"../data/{args.data_set}/{args.data_set}_genes_ordered_by_variance.pkl"

    print(f"Loading AnnData: {DATA_H5AD_PATH}")
    adata = ad.read_h5ad(DATA_H5AD_PATH)

    print("Loading gene list")
    gene_list = load_gene_list(GENE_NAME_PATH)
    gene_list_upper = [g.upper() for g in gene_list]

    # ----------------------------------
    # Filter genes (order preserved)
    # ----------------------------------
    if "gene_symbol" in adata.var.columns:
        var_genes = adata.var["gene_symbol"].astype(str).str.upper().values
    else:
        var_genes = adata.var_names.astype(str).str.upper().values

    gene_to_idx = {g: i for i, g in enumerate(var_genes)}

    idx = [gene_to_idx[g] for g in gene_list_upper if g in gene_to_idx]
    adata = adata[:, idx].copy()

    print(f"After filtering: {adata.shape}")

    # ----------------------------------
    # IDs to save by
    # ----------------------------------
    if args.data_set == "TARGET_AML":
        ids = adata.obs_names.astype(str).tolist()   # aliquot/sample IDs like ...-09A
    else:
        obs_col = "cases.submitter_id" if args.data_set == "TCGA" else "patient_id"
        ids = adata.obs.loc[:, obs_col].astype(str).tolist()

    print(f"Computing embeddings for {len(ids)} patients/samples")
    print("Sample IDs:", ids[:5])

    # ----------------------------------
    # Dense matrix
    # ----------------------------------
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # ----------------------------------
    # PCA 50 and 100
    # ----------------------------------
    base_out = Path(f"../data/{args.data_set}/PCA")

    for dim in [50, 100]:

        print(f"\nRunning PCA dim={dim}")
        pca = PCA(n_components=dim, random_state=0)
        Z = pca.fit_transform(X)

        out_dir = base_out / f"PCA_{dim}"
        save_per_sample_embeddings(Z, ids, out_dir)

        print(f"Saved PCA {dim} embeddings → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, required=True)
    args = parser.parse_args()
    main(args)
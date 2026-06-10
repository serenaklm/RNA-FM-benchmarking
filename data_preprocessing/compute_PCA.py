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

    # Allow explicit overrides for non-standard datasets (e.g. ICI cohorts)
    if args.data_dir:
        DATA_H5AD_PATH = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
        base_out = Path(args.data_dir) / "PCA"
    else:
        DATA_H5AD_PATH = f"../data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"
        base_out = Path(f"../data/{args.data_set}/PCA")

    GENE_NAME_PATH = args.gene_name_path if args.gene_name_path else \
        f"../data/{args.data_set}/{args.data_set}_genes_ordered_by_variance.pkl"

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
    elif args.data_set == "TCGA":
        ids = adata.obs.loc[:, "cases.submitter_id"].astype(str).tolist()
    elif "patient_id" in adata.obs.columns:
        ids = adata.obs.loc[:, "patient_id"].astype(str).tolist()
    else:
        # Fallback for ICI cohorts where obs_names are the patient IDs
        ids = adata.obs_names.astype(str).tolist()

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

    for dim in [50, 100]:
        actual_dim = min(dim, X.shape[0] - 1)
        if actual_dim < dim:
            print(f"\nRunning PCA dim={actual_dim} (capped from {dim}: only {X.shape[0]} samples)")
        else:
            print(f"\nRunning PCA dim={dim}")
        pca = PCA(n_components=actual_dim, random_state=0)
        Z = pca.fit_transform(X)

        out_dir = base_out / f"PCA_{dim}"
        save_per_sample_embeddings(Z, ids, out_dir)

        print(f"Saved PCA {dim} embeddings → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, required=True)
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override data directory (e.g. for ICI cohorts). "
             "If set, h5ad is read from {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad "
             "and PCA is saved to {data_dir}/PCA/.",
    )
    parser.add_argument(
        "--gene_name_path", type=str, default=None,
        help="Override gene list pickle path (default: inferred from --data_set). "
             "Use TCGA gene list for ICI cohorts.",
    )
    args = parser.parse_args()
    main(args)
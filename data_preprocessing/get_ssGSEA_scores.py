"""
Compute ssGSEA enrichment scores for RNA-seq data
Takes gene set name as argument (immuno, aurora, hallmark, combined)
"""

import os
import pandas as pd
import numpy as np
import anndata as ad
import argparse
from gseapy import ssgsea as gp_ssgsea
from gseapy.parser import read_gmt
import pickle

def process_rna_seq(RNA_seq_info, gene_names) -> pd.DataFrame:
    """
    Load and process an AnnData RNA-seq file into a dense samples×genes DataFrame.

    Steps:
    1. Convert the expression matrix to a dense DataFrame.
    2. Replace column names with gene symbols.
    3. Collapse duplicate gene symbols (keep max per row).
    4. Filter to a specified gene list.
    """
    
    # Convert to dense DataFrame
    X = RNA_seq_info.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    elif hasattr(X, "A"):
        X = X.A
    
    # Build DataFrame
    RNA_seq = pd.DataFrame(X, index=RNA_seq_info.obs_names)
    RNA_seq.columns = RNA_seq_info.var["gene_symbol"].astype(str)
    
    # Collapse duplicates by gene symbol
    RNA_seq = RNA_seq.groupby(RNA_seq.columns, axis=1).max()
    
    # Filter to desired gene list (order preserved)
    RNA_seq = RNA_seq.loc[:, [g for g in gene_names.keys() if g in RNA_seq.columns]]
    
    return RNA_seq

def _to_dense_df(adata) -> pd.DataFrame:
    """Return a dense spots×genes DataFrame with UPPERCASE gene symbols."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    elif hasattr(X, "A"):
        X = X.A
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df.columns = df.columns.astype(str).str.upper()
    return df

def _read_gmt(gmt_path: str) -> dict:
    """Load a GMT as { set_name: [GENE, ...] } with UPPERCASE symbols."""
    lib = read_gmt(gmt_path)
    out = {}
    if isinstance(lib, dict):
        for k, v in lib.items():
            out[str(k)] = [g.strip().upper() for g in v if g]
    else:
        # Tuple/list variants from older gseapy: (name, desc, genes) or (name, genes)
        for item in lib:
            if len(item) >= 3:
                name, genes = item[0], item[2]
            elif len(item) == 2:
                name, genes = item[0], item[1]
            else:
                continue
            out[str(name)] = [g.strip().upper() for g in genes if g]
    return out

def _run_ssgsea(expr_patients_x_genes: pd.DataFrame, gene_sets: dict) -> pd.DataFrame:
    """
    Run ssGSEA via gseapy.
    Input:  expr (patients x genes), gene_sets {set: [genes]}
    Output: scores (patients x sets)
    """
    res = gp_ssgsea(
        data=expr_patients_x_genes.T,        # gseapy expects genes x samples
        gene_sets=gene_sets,
        outdir=None,
        min_size=1,
        max_size=5000,
        sample_norm_method="rank",
        permutation_num=0,
        no_plot=True,
        scale=False,
        processes=1,
        format="png",
    )

    # gseapy versions differ: prefer .scores if present; else pivot res2d (NES)
    if hasattr(res, "scores") and isinstance(res.scores, pd.DataFrame):
        df = res.scores  # samples x sets
    elif hasattr(res, "res2d") and isinstance(res.res2d, pd.DataFrame):
        df = res.res2d.pivot(index="Name", columns="Term", values="NES")
    else:
        raise RuntimeError("Unexpected gseapy.ssgsea return structure.")

    df = df.reindex(expr_patients_x_genes.index)
    df = df.replace([np.inf, -np.inf], np.nan)  # FutureWarning is harmless
    return df

def main(args):
    gene_set_files = {
        "immuno": "immuno_gene_sets.gmt",
        "aurora": "AURORA_gene_sets.gmt",
        "hallmark": "50_hallmark_gene_sets.gmt",
        "combined": "combined_gene_sets.gmt",
    }

    # -------------------------
    # Parse which gene sets to run
    # -------------------------
    if args.gene_set.lower() == "all":
        gene_sets_to_run = list(gene_set_files.keys())
    else:
        gene_sets_to_run = [s.strip().lower() for s in args.gene_set.split(",") if s.strip()]
        bad = [s for s in gene_sets_to_run if s not in gene_set_files]
        if bad:
            raise ValueError(
                f"Unknown gene_set(s): {bad}. Allowed: {list(gene_set_files.keys())} or 'all' "
                f"or comma-list like 'immuno,aurora'."
            )

    # Allow explicit overrides for non-standard datasets (e.g. ICI cohorts)
    if args.data_dir:
        DATA_H5AD_PATH = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
    else:
        DATA_H5AD_PATH = f"../data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"

    GMT_DIR = "../data/gene_sets/processed"

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = f"../data/{args.data_set}/ssGSEA_scores/"

    if args.gene_name_path:
        GENE_NAME_PATH = args.gene_name_path
    else:
        GENE_NAME_PATH = f"../data/{args.data_set}/{args.data_set}_genes_ordered_by_variance.pkl"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"Starting ssGSEA enrichment score computation for {args.data_set}")
    print(f"Gene sets to run: {gene_sets_to_run}")
    print("=" * 70)

    # -------------------------
    # Load dataset + build expression ONCE
    # -------------------------
    print(f"\nLoading {args.data_set} data from: {DATA_H5AD_PATH}")
    adata = ad.read_h5ad(DATA_H5AD_PATH)
    print(f"Loaded: {adata.shape[0]} patients, {adata.shape[1]} genes")

    with open(GENE_NAME_PATH, "rb") as f:
        gene_names = pickle.load(f)

    print("\nConverting to dense DataFrame...")
    expr = process_rna_seq(adata, gene_names)          # samples x genes
    expr = expr.loc[:, expr.std(axis=0) > 0]           # drop constant genes
    print(f"Expression matrix after removing constant genes: {expr.shape}")

    # -------------------------
    # Loop gene sets and run ssGSEA
    # -------------------------
    for gs_name in gene_sets_to_run:
        output_pkl = os.path.join(OUTPUT_DIR, f"{gs_name}.pkl")
        if args.skip_existing and os.path.exists(output_pkl):
            print(f"\n[SKIP] {gs_name}: already exists at {output_pkl}")
            continue

        print("\n" + "=" * 70)
        print(f"Loading gene sets: {gs_name}")
        print("=" * 70)

        gmt_path = os.path.join(GMT_DIR, gene_set_files[gs_name])
        print(f"Loading: {os.path.basename(gmt_path)}")
        gene_sets = _read_gmt(gmt_path)
        print(f"Loaded {len(gene_sets)} gene sets")

        print("\n" + "=" * 70)
        print(f"Computing ssGSEA scores: {gs_name}")
        print("=" * 70)

        scores_df = _run_ssgsea(expr, gene_sets)  # samples x sets
        scores_df = scores_df.replace([np.inf, -np.inf], np.nan)

        print(f"ssGSEA complete for {gs_name}. Output shape: {scores_df.shape}")

        scores_df.to_pickle(output_pkl)
        print(f"Saved pickle to: {output_pkl}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gene_set",
        type=str,
        required=True,
        help="Gene set(s): immuno|aurora|hallmark|combined, "
             "or comma-list (e.g. immuno,aurora), or 'all'.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip gene sets whose output pickle already exists.",
    )
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        help="Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...).",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override data directory (e.g. for ICI cohorts). "
             "If set, h5ad is read from {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad.",
    )
    parser.add_argument(
        "--gene_name_path", type=str, default=None,
        help="Override gene list pickle path. Use the TCGA gene list for ICI cohorts.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory for ssGSEA score pickles.",
    )
    args = parser.parse_args()
    main(args)

"""
Pre-tokenize RNA-seq data for BulkFormer fine-tuning.

Follows get_BulkFormer_embeddings.py exactly for preprocessing:
  1. Load log1p h5ad and build expression DataFrame with gene symbols.
  2. Map gene symbols → Ensembl IDs via mygene API.
  3. Align to BulkFormer gene list (missing genes filled with -10).
  4. Save per-patient float32 array of shape (20010,) — same input the model receives.

Run with bulkformer conda env from the RNA-FM-benchmarking root:
  cd data_preprocessing/tokenize_for_finetuning/
  python BulkFormer_tokenize.py --data_set TCGA
"""

import os
import sys
import pickle
import argparse
import yaml
from pathlib import Path

import mygene
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
_data_paths = yaml.safe_load(open(REPO_ROOT / "configs" / "data_paths.yaml"))
# BulkFormer data files (gene list)
BULKFORMER_DIR = _data_paths["model_repos"]["BulkFormer"]

# ---------------------------------------------------------------------------
# Utility functions — copied verbatim from get_BulkFormer_embeddings.py
# ---------------------------------------------------------------------------

def symbols_to_ensg(symbols, species="human"):
    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        symbols,
        scopes="symbol,alias,prev_symbol",
        fields="ensembl.gene,symbol,name",
        species=("9606" if species == "human" else "10090"),
        as_dataframe=True,
        returnall=False,
    )
    df = hits.reset_index().rename(columns={"query": "symbol"})

    def pick_ensg(x):
        if isinstance(x, list):
            for v in x:
                gid = v.get("gene") if isinstance(v, dict) else v
                if gid and gid.startswith(("ENSG", "ENSMUSG")):
                    return gid.split(".")[0]
            return pd.NA
        if isinstance(x, dict):
            gid = x.get("gene")
            return gid.split(".")[0] if gid else pd.NA
        if isinstance(x, str):
            return x.split(".")[0]
        return pd.NA

    df["ensg_id"] = df["ensembl.gene"].map(pick_ensg)
    out = df[["symbol", "ensg_id", "name"]].drop_duplicates(subset=["symbol"])
    return out


def main_gene_selection(X_df, gene_list):
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    padding_df = pd.DataFrame(
        np.full((X_df.shape[0], len(to_fill_columns)), -10),
        columns=to_fill_columns,
        index=X_df.index,
    )

    X_df = pd.DataFrame(
        np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
        index=X_df.index,
        columns=list(X_df.columns) + list(padding_df.columns),
    )
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var["mask"] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ---- paths ----
    data_path = f"../../data/{dataset}/RNA/{dataset}_bulk_RNA_processed_log1p.h5ad"
    gene_info_path = os.path.join(BULKFORMER_DIR, "data/bulkformer_gene_info.csv")
    output_dir = "../../data/tokens/BulkFormer"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_tokens.pkl")

    # ---- load BulkFormer gene list ----
    bulkformer_gene_info = pd.read_csv(gene_info_path)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].tolist()
    print(f"BulkFormer gene list size: {len(bulkformer_gene_list)}")

    # ---- load RNA data ----
    print(f"Reading {dataset} data...")
    data = anndata.read_h5ad(data_path)

    # ---- patient IDs (same logic as embedding script) ----
    if dataset == "TARGET_AML":
        sample_ids = data.obs_names.astype(str).tolist()
    else:
        obs_key = "cases.submitter_id" if dataset == "TCGA" else "patient_id"
        sample_ids = data.obs[obs_key].astype(str).tolist()

    # ---- build expression DataFrame with gene symbols as columns ----
    data_df = pd.DataFrame(
        data.X,
        index=sample_ids,
        columns=data.var["gene_symbol"].values,
    )
    print(f"Data shape before sieving: {data_df.shape}")

    # ---- map gene symbols → Ensembl IDs ----
    print("Mapping gene symbols to Ensembl IDs via mygene (may take a moment)...")
    mapping = symbols_to_ensg(data_df.columns)
    mapping["ensg_id"] = mapping["ensg_id"].replace({"<NA>": pd.NA})
    mapping = mapping.dropna(subset=["ensg_id"])
    mapping = mapping[mapping["ensg_id"].isin(bulkformer_gene_list)]

    data_df_sieved = data_df.loc[:, [c[0] for c in mapping["symbol"].values.tolist()]]
    data_df_sieved.columns = mapping["ensg_id"]
    data_df_sieved = data_df_sieved.groupby(data_df_sieved.columns, axis=1).mean()
    print(f"Data shape after sieving: {data_df_sieved.shape}")

    # ---- align to BulkFormer gene list (missing genes → -10) ----
    input_df, to_fill_columns, _ = main_gene_selection(
        X_df=data_df_sieved, gene_list=bulkformer_gene_list
    )
    n_present = len(bulkformer_gene_list) - len(to_fill_columns)
    print(f"Genes present: {n_present} / {len(bulkformer_gene_list)} "
          f"({n_present / len(bulkformer_gene_list):.1%})")
    print(f"Genes filled with -10 (missing): {len(to_fill_columns)}")

    # ---- save per-patient float32 arrays ----
    expr_array = input_df.values.astype(np.float32)  # (N, 20010)

    tokens_dict = {}
    for i, pid in tqdm(enumerate(sample_ids), total=len(sample_ids), desc="Saving"):
        tokens_dict[pid] = expr_array[i]  # shape (20010,)

    save_pickle(tokens_dict, output_path)
    print(f"Saved {len(tokens_dict)} samples → {output_path}")
    print(f"Array shape per patient: {expr_array.shape[1:]}")

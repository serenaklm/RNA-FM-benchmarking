"""
Pre-process RNA-seq data for COMPASS fine-tuning.

Replicates the preprocessing done inside COMPASS.py (get_FM_embeddings):
  1. Build DataFrame with cancer_code (int) as column 0 + gene expression columns.
  2. Apply pretrainer.scaler.transform():
       - column 0 (cancer_code) is left unchanged
       - gene columns: log2(x+1) then MinMaxScaler to [0,1]
  3. Save per-patient float32 array of shape (1 + n_compass_genes,).

The first element of each array is cancer_code (as float32, cast to long by the encoder).
The remaining elements are the scaled log2 gene expression values.

Run with compass conda env:
  cd data_preprocessing/tokenize_for_finetuning/
  python COMPASS_preprocess.py --data_set TCGA
"""

import os
import sys
import pickle
import argparse

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "/data/rbg/users/ashyamal/COMPASS")
from compass import loadcompass


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# Cancer type → COMPASS integer code (from immuno-compass.com/download/other/cancer_code.json)
CANCER_MAPPING = {
    "ACC": 0, "BLCA": 1, "BRCA": 2, "CESC": 3, "CHOL": 4, "COAD": 5,
    "DLBC": 6, "ESCA": 7, "GBM": 8, "HNSC": 9, "KICH": 10, "KIRC": 11,
    "KIRP": 12, "LAML": 13, "LGG": 14, "LIHC": 15, "LUAD": 16, "LUSC": 17,
    "MESO": 18, "OV": 19, "PAAD": 20, "PCPG": 21, "PRAD": 22, "READ": 23,
    "SARC": 24, "SKCM": 25, "STAD": 26, "TGCT": 27, "THCA": 28, "THYM": 29,
    "UCEC": 30, "UCS": 31, "UVM": 32,
}


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

    # ---- paths (relative to this script's location) ----
    adata_paths = {
        "TCGA":       "../../data/TCGA/RNA/TCGA_bulk_RNA_raw.h5ad",
        "CALGB":      "../../data/CALGB/RNA/CALGB_bulk_RNA_processed.h5ad",
        "CPTAC":      "../../data/CPTAC/RNA/CPTAC_bulk_RNA_processed.h5ad",
        "Hartwig":    "../../data/Hartwig/RNA/Hartwig_bulk_RNA_processed.h5ad",
        "ICGC":       "../../data/ICGC/RNA/ICGC_bulk_RNA_processed.h5ad",
        "TARGET_AML": "../../data/TARGET_AML/RNA/TARGET_AML_bulk_RNA_processed.h5ad",
    }
    cancer_info_paths = {
        "CPTAC":   "../../data/CPTAC/obs_to_tcga_mapping.pkl",
        "ICGC":    "../../data/ICGC/id_to_tcga_mapping.pkl",
        "Hartwig": "../../data/Hartwig/id_to_tcga_mapping.pkl",
    }
    compass_gene_map_path = "../../data/COMPASS/compass_gene_map.csv"
    pretrainer_path = "../../data/COMPASS/pretrainer.pt"
    output_dir = "../../data/tokens/COMPASS"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_tokens.pkl")

    # ---- load COMPASS pretrainer (for scaler) ----
    print("Loading COMPASS pretrainer (for scaler)...")
    pretrainer = loadcompass(pretrainer_path)
    scaler = pretrainer.scaler

    # ---- load gene map ----
    gene_map = pd.read_csv(compass_gene_map_path)
    compass_genes = gene_map["gene_name"].tolist()
    print(f"COMPASS gene vocab size: {len(compass_genes)}")

    # ---- load RNA data ----
    print(f"Reading {dataset} data...")
    adata = ad.read_h5ad(adata_paths[dataset])

    # ---- cancer code column ----
    if dataset in cancer_info_paths:
        id_to_tcga = pickle.load(open(cancer_info_paths[dataset], "rb"))
        obs_to_use = set(id_to_tcga.keys())
        print(f"Filtering {adata.n_obs} → in cancer-type mapping")
        adata = adata[adata.obs["patient_id"].isin(obs_to_use)].copy()
        print(f"After filter: {adata.n_obs} patients")

    # ---- patient IDs ----
    if dataset == "TARGET_AML":
        ids = adata.obs_names.astype(str).tolist()
    else:
        obs_col = "cases.submitter_id" if dataset == "TCGA" else "patient_id"
        ids = adata.obs[obs_col].astype(str).tolist()

    # ---- build base DataFrame (same as COMPASS.py) ----
    df = pd.DataFrame(
        adata.X,
        index=ids,
        columns=adata.var["gene_symbol"].values,
    )
    df.insert(0, "bcr_patient_barcode", ids)

    if dataset == "TCGA":
        cancer_types = adata.obs["project.project_id"].str.replace("TCGA-", "", regex=False)
        df.insert(1, "cancer_code", cancer_types.map(CANCER_MAPPING).values)
    elif dataset == "CALGB":
        df.insert(1, "cancer_code", 2)  # BRCA
    elif dataset in cancer_info_paths:
        cancer_types = adata.obs["patient_id"].map(id_to_tcga)
        df.insert(1, "cancer_code", cancer_types.map(CANCER_MAPPING).values)
    elif dataset == "TARGET_AML":
        df.insert(1, "cancer_code", 13)  # LAML

    # ---- align to compass gene vocab ----
    target_columns = ["bcr_patient_barcode", "cancer_code"] + compass_genes
    df_aligned = df.reindex(columns=target_columns, fill_value=0.0)
    missing = set(compass_genes) - set(df.columns)
    print(f"Zero-filled {len(missing)} missing genes out of {len(compass_genes)}")
    df_for_compass = df_aligned.set_index("bcr_patient_barcode")

    print(f"Input shape: {df_for_compass.shape}")

    # ---- scale: cancer_code unchanged, genes → log2(x+1) → MinMax ----
    print("Applying COMPASS scaler...")
    df_scaled = scaler.transform(df_for_compass)

    # ---- save per-patient float32 arrays ----
    gene_cols = [c for c in df_scaled.columns if c != "cancer_code"]
    tokens_dict = {}
    for i, pid in tqdm(enumerate(ids), desc="Saving", total=len(ids)):
        row = df_scaled.iloc[i]  # positional indexing avoids duplicate-ID issues
        cancer_code = float(row["cancer_code"])
        expr = row[gene_cols].values.astype(np.float32)
        # shape: (1 + n_compass_genes,) — matches TransformerEncoder input
        tokens_dict[pid] = np.concatenate([[cancer_code], expr]).astype(np.float32)

    save_pickle(tokens_dict, output_path)

    arr = np.stack(list(tokens_dict.values()))
    print(f"Saved {len(tokens_dict)} samples to {output_path}")
    print(f"Array shape per patient: {arr.shape[1:]}")
    print(f"Expression stats (genes only): min={arr[:, 1:].min():.3f}, "
          f"max={arr[:, 1:].max():.3f}, mean={arr[:, 1:].mean():.4f}")

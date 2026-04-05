import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pickle
import anndata
from anndata import AnnData
import anndata as ad
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

def pickle_data(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path): 

    with open(path, "rb") as f: 
        return pickle.load(f)

def list_patient_files(raw_dir: str) -> list[Path]:
    files = sorted(Path(raw_dir).glob("*.isf.transcript_data.csv"))
    if not files:
        raise FileNotFoundError(f"No *.isf.transcript_data.csv files found in {raw_dir}")
    return files

def patient_id_from_filename(fp: Path) -> str:
    return fp.name.replace(".isf.transcript_data.csv", "")


def load_symbol_level_tpm_filtered(
    csv_path: Path,
    tpm_col: str = "AdjTPM",
) -> pd.Series:
    """
    Returns Series indexed by gene SYMBOL, summing TPM across transcripts,
    filtered to allowed_symbols.
    """
    df = pd.read_csv(csv_path, usecols=["GeneName", tpm_col])
    df = df.dropna(subset=["GeneName"]).copy()
    df["GeneName"] = df["GeneName"].astype(str)

    if df.empty:
        return pd.Series(dtype=float)

    sym_tpm = df.groupby("GeneName", sort=False)[tpm_col].sum()
    sym_tpm = pd.to_numeric(sym_tpm, errors="coerce").fillna(0.0)
    return sym_tpm

def build_hartwig_filtered_anndata_unordered(
    raw_dir: str,
    tpm_col: str = "AdjTPM",
    verbose_every: int = 25,
) -> ad.AnnData:
    files = list_patient_files(raw_dir)

    print(f"Found {len(files)} patients")
    print(f"Using TPM column: {tpm_col}")

    # Pass 1: read each patient, store filtered series, collect union genes
    patient_ids = []
    series_by_patient = []
    gene_union = set()

    for i, fp in tqdm(enumerate(files)):
        pid = patient_id_from_filename(fp)
        patient_ids.append(pid)

        s = load_symbol_level_tpm_filtered(fp, tpm_col=tpm_col)
        series_by_patient.append(s)
        gene_union.update(s.index.tolist())

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[{i+1}/{len(files)}] processed; genes matched this sample: {len(s)}")

    gene_list = sorted(gene_union)  # ordering arbitrary
    print(f"Genes retained (TCGA and Hartwig): {len(gene_list)}")

    sym_to_col = {g: j for j, g in enumerate(gene_list)}
    X = np.zeros((len(files), len(gene_list)), dtype=np.float32)

    for i, s in enumerate(series_by_patient):
        row = X[i]
        for sym, val in s.items():
            row[sym_to_col[sym]] = float(val)

    obs = pd.DataFrame({"patient_id": patient_ids}, index=patient_ids)
    var = pd.DataFrame(index=gene_list)
    var["gene_symbol"] = gene_list

    # log_variance on log1p(TPM)
    X_log = np.log1p(X)
    var["log_variance"] = np.var(X_log, axis=0)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names.name = "patient_id"
    adata.var_names.name = "gene_symbol"

    print(f"Final shape: {adata.shape}")
    return adata

if __name__ == "__main__":
    # Construct adata
    print("Constructing AnnData from Hartwig raw files...")
    raw_dir = "/data/rbg/shared/datasets/Hartwig/isofox_tpm_update_4/raw"

    adata = build_hartwig_filtered_anndata_unordered(
        raw_dir=raw_dir,
        tpm_col="AdjTPM",
    )
    
    # Filter for genes with mean expression over 0.1
    mean_expression = adata.X.mean(axis=0)
    genes_to_keep = mean_expression > 0.1

    # how many genes to keep
    print(f"Number of genes before filtering: {adata.n_vars}")
    print(f"Number of genes after filtering: {genes_to_keep.sum()}")

    filtered_adata = adata[:, genes_to_keep]

    print(f"Constructed AnnData with shape {filtered_adata.shape}")
    print(filtered_adata)
    
    filtered_adata.write_h5ad("/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/Hartwig/RNA/Hartwig_bulk_RNA_processed.h5ad")
    print("Saved untransformed AnnData to h5ad.")
    
    # Construct log-transformed filtered_adata
    X_log = np.log1p(filtered_adata.X)
    log_adata = ad.AnnData(X=X_log, obs=filtered_adata.obs.copy(), var=filtered_adata.var.copy())
    
    print(f"Constructed log-transformed AnnData with shape {log_adata.shape}")
    print(log_adata)
    
    log_adata.write_h5ad("/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/Hartwig/RNA/Hartwig_bulk_RNA_processed_log1p.h5ad")
    print("Saved log-transformed AnnData to h5ad.")
    
    
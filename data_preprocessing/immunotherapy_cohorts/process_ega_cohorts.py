"""
process_ega_cohorts.py

Process IMMotion150 and IMVigor210 bulk RNA-seq TPM files into AnnData objects,
matching the pipeline used for Hugo/Riaz/Rose in process_ICI_cohorts.ipynb.

TPM files are STAR output (GENCODE v36) located at:
  data/immunotherapy_cohorts/{dataset}/RNA/paired/{sample_id}/tpm_{sample_id}.tsv

Outputs:
  data/immunotherapy_cohorts/{dataset}/RNA/{dataset}_bulk_RNA_processed.h5ad
  data/immunotherapy_cohorts/{dataset}/RNA/{dataset}_bulk_RNA_processed_log1p.h5ad

Usage (from repo root):
  python data_preprocessing/immunotherapy_cohorts/process_ega_cohorts.py
  python data_preprocessing/immunotherapy_cohorts/process_ega_cohorts.py --dataset IMMotion150
  python data_preprocessing/immunotherapy_cohorts/process_ega_cohorts.py --dataset IMVigor210
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data" / "immunotherapy_cohorts"
MIN_MEAN_TPM = 0.1

# GENCODE uses "protein_coding" (underscore); NCBI uses "protein-coding" (hyphen)
PROTEIN_CODING_TAG = "protein_coding"

# ── Per-cohort configuration ───────────────────────────────────────────────────
COHORT_CONFIGS = {
    "IMMotion150": {
        "cancer_type": "KIRC",
        "manifest_sample_col": "AnnoRNASampleID",
        "response_col": "BestResponse",
        "arm_col": "ARM",
        "arm_to_ici": {
            "Atezo+Bev": "Atezolizumab+Bevacizumab",
            "Atezo":     "Atezolizumab",
            "Sunitinib": "Sunitinib",
        },
        "arm_to_ici_target": {
            "Atezo+Bev": "PD-L1",
            "Atezo":     "PD-L1",
            "Sunitinib": None,
        },
        "extra_obs": ["ARM", "PFS", "CNSR_PFS", "MSKCC", "Stage"],
    },
    "IMVigor210": {
        "cancer_type": "BLCA",
        "manifest_sample_col": "sample_alias",
        "response_col": "Best Confirmed Overall Response",
        "arm_col": None,
        "ici": "Atezolizumab",
        "ici_target": "PD-L1",
        "extra_obs": [
            "os", "censOS", "IC Level", "TC Level", "Immune phenotype",
            "Lund2", "TCGA Subtype", "FMOne mutation burden per MB",
            "Baseline ECOG Score", "Received platinum", "Met Disease Status", "Tissue",
        ],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest(dataset: str, cfg: dict) -> pd.DataFrame:
    """Load the download manifest, deduplicated to one row per sample."""
    path = DATA_DIR / dataset / "RNA" / "rna_seq_download_manifest.csv"
    df = pd.read_csv(path)
    sample_col = cfg["manifest_sample_col"]
    df = df.drop_duplicates(subset=[sample_col]).set_index(sample_col)
    return df


def load_tpm_file(tpm_path: Path) -> pd.DataFrame:
    """
    Load a single STAR TPM TSV. Returns a DataFrame indexed by gene_name with
    columns [tpm_unstranded, gene_type]. Duplicate gene symbols are collapsed
    by summing TPM; gene_type takes the first value.
    """
    df = pd.read_csv(
        tpm_path, sep="\t", comment="#",
        usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"],
    )
    # Drop STAR summary rows (N_unmapped, N_multimapping, N_noFeature, N_ambiguous)
    df = df[~df["gene_id"].str.startswith("N_")]
    df = df.dropna(subset=["gene_name"])
    df = df.groupby("gene_name", as_index=True).agg(
        tpm_unstranded=("tpm_unstranded", "sum"),
        gene_type=("gene_type", "first"),
    )
    return df


def build_expression_matrix(
    paired_dir: Path, sample_ids: list
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load all per-sample TPM files and assemble a (samples x genes) DataFrame.
    Returns (expr_df, gene_type_series).
    """
    tpm_dict = {}
    gene_type_series = None
    missing = []

    for sid in tqdm(sample_ids, desc="Loading TPM files"):
        tpm_path = paired_dir / sid / f"tpm_{sid}.tsv"
        if not tpm_path.exists():
            missing.append(sid)
            continue
        parsed = load_tpm_file(tpm_path)
        tpm_dict[sid] = parsed["tpm_unstranded"]
        if gene_type_series is None:
            gene_type_series = parsed["gene_type"]

    if missing:
        print(f"\n  WARNING: {len(missing)} sample(s) have no TPM file and will be excluded:")
        for sid in missing:
            print(f"    {sid}")
    else:
        print("  All TPM files found.")

    expr_df = pd.DataFrame(tpm_dict).T   # samples x genes
    expr_df.index.name = "sample_id"
    return expr_df, gene_type_series


def make_response_label(val) -> str | None:
    """CR/PR → 'R', SD/PD → 'NR', NE/missing → None."""
    if pd.isna(val):
        return None
    v = str(val).strip().upper()
    if v in ("CR", "PR"):
        return "R"
    if v in ("SD", "PD"):
        return "NR"
    return None


def process_adata(adata: ad.AnnData, min_mean_tpm: float = MIN_MEAN_TPM):
    """
    Filter low-expression and non-protein-coding genes; return
    (processed_raw_tpm, processed_log1p) matching process_ICI_cohorts.ipynb.
    """
    cohort = adata.uns.get("cohort", "?")

    gene_means = np.mean(adata.X, axis=0)
    adata = adata[:, gene_means >= min_mean_tpm].copy()
    print(f"[{cohort}] After mean TPM >= {min_mean_tpm} filter: {adata.n_vars} genes")

    if "gene_type" in adata.var.columns:
        adata = adata[:, adata.var["gene_type"] == PROTEIN_CODING_TAG].copy()
        print(f"[{cohort}] After protein-coding filter: {adata.n_vars} genes")

    adata.uns["processing_steps"] = [
        f"filtered_low_expression_mean_tpm_{min_mean_tpm}",
        "filtered_protein_coding",
    ]

    adata_log1p = adata.copy()
    adata_log1p.X = np.log1p(adata_log1p.X)
    adata_log1p.var["log_variance"] = np.var(adata_log1p.X, axis=0)
    adata_log1p.uns["processing_steps"] = adata.uns["processing_steps"] + [
        "log1p_transformed", "computed_log_variance",
    ]

    print(f"[{cohort}] Final: {adata.n_obs} x {adata.n_vars}")
    return adata, adata_log1p


def process_cohort(dataset: str):
    cfg = COHORT_CONFIGS[dataset]
    paired_dir = DATA_DIR / dataset / "RNA" / "paired"
    out_dir    = DATA_DIR / dataset / "RNA"

    print(f"\n{'='*60}")
    print(f" Processing: {dataset}")
    print(f"{'='*60}")

    # 1. Manifest → one row per sample
    manifest = load_manifest(dataset, cfg)
    all_sample_ids = manifest.index.tolist()
    print(f"Manifest samples: {len(all_sample_ids)}")

    # 2. Expression matrix
    expr_df, gene_type_series = build_expression_matrix(paired_dir, all_sample_ids)
    print(f"Expression matrix: {expr_df.shape}  (samples x genes)")

    # 3. obs DataFrame
    obs = manifest.loc[expr_df.index].copy()

    obs["response_label"] = obs[cfg["response_col"]].apply(make_response_label)
    obs["cancer_type"] = cfg["cancer_type"]
    obs["Timing"] = "Pre"

    if cfg["arm_col"] and cfg["arm_col"] in obs.columns:
        obs["ICI"]        = obs[cfg["arm_col"]].map(cfg["arm_to_ici"])
        obs["ICI_target"] = obs[cfg["arm_col"]].map(cfg["arm_to_ici_target"])
    else:
        obs["ICI"]        = cfg.get("ici")
        obs["ICI_target"] = cfg.get("ici_target")

    # Keep only columns that exist in the manifest
    keep_extra = [c for c in cfg["extra_obs"] if c in obs.columns]
    obs = obs[["response_label", "cancer_type", "Timing", "ICI", "ICI_target"] + keep_extra]

    # 4. Filter to labeled samples (drop NE / missing response)
    n_before = len(obs)
    obs = obs[obs["response_label"].notna()]
    expr_df = expr_df.loc[obs.index]
    n_dropped = n_before - len(obs)
    print(f"Samples after response filter: {len(obs)}  (dropped {n_dropped} NE/missing)")
    print(f"  R: {(obs['response_label'] == 'R').sum()}  "
          f"NR: {(obs['response_label'] == 'NR').sum()}")

    # 5. var DataFrame
    var = pd.DataFrame(index=expr_df.columns)
    var.index.name = "gene_symbol"
    var["gene_symbol"] = var.index
    if gene_type_series is not None:
        var["gene_type"] = var.index.map(gene_type_series.to_dict())

    # 6. Raw AnnData
    adata_raw = ad.AnnData(
        X=expr_df.values.astype("float32"),
        obs=obs,
        var=var,
    )
    adata_raw.uns["cohort"] = dataset
    print(f"Raw AnnData: {adata_raw.n_obs} x {adata_raw.n_vars}")

    # 7. Process and save
    adata_processed, adata_log1p = process_adata(adata_raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    adata_processed.write_h5ad(out_dir / f"{dataset}_bulk_RNA_processed.h5ad")
    adata_log1p.write_h5ad(out_dir / f"{dataset}_bulk_RNA_processed_log1p.h5ad")

    print(f"Saved:")
    print(f"  {out_dir / f'{dataset}_bulk_RNA_processed.h5ad'}")
    print(f"  {out_dir / f'{dataset}_bulk_RNA_processed_log1p.h5ad'}")


def main():
    parser = argparse.ArgumentParser(
        description="Process IMMotion150 and IMVigor210 TPM files into AnnData."
    )
    parser.add_argument(
        "--dataset",
        choices=list(COHORT_CONFIGS.keys()),
        default=None,
        help="Cohort to process. Processes both if omitted.",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(COHORT_CONFIGS.keys())
    for ds in datasets:
        process_cohort(ds)

    print("\nDone.")


if __name__ == "__main__":
    main()

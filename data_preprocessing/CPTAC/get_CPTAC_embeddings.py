#!/usr/bin/env python3
print("Importing...")

import argparse
import json
import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any

# Import from data_preprocessing package
from ..preprocess_utils import (
    load_foundation_model_embeddings,
    load_ssgsea_scores,
    compute_max_mean_gene_set_features,
    ensure_genes_present,
    get_embedding_for_sample,
    load_pca_embeddings,
    get_pca_for_sample,
    pickle_data,
    load_pickle,
)

print("Finished imports")


# -------------------------
# CPTAC clinical helpers
# -------------------------
def add_pfs(df: pd.DataFrame, id_col: str = "cases.case_id") -> pd.DataFrame:
    """
    Compute PFS_time and PFS_event per patient (case_id) from a long DF
    containing multiple follow_ups rows.

    Returns a DataFrame with:
        id_        (the case id)
        PFS_time   (float, days)
        PFS_event  (0/1)
    """
    def _compute_pfs(g: pd.DataFrame) -> pd.Series:
        # candidate event times
        event_times = []
        for col in [
            "follow_ups.days_to_progression",
            "follow_ups.days_to_recurrence",
            "follow_ups.days_to_first_event",
        ]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                if not vals.empty:
                    event_times.append(vals.min())
        t_prog = np.nan if len(event_times) == 0 else min(event_times)

        # progression flag
        prog_flag = False
        prog_col = "follow_ups.progression_or_recurrence"
        if prog_col in g.columns:
            prog_flag = g[prog_col].astype(str).str.upper().isin(
                ["YES", "Y", "PROGRESSION", "RECURRENCE"]
            ).any()

        # censor time candidates
        censor_candidates = []
        for col in [
            "follow_ups.days_to_follow_up",
            "follow_ups.days_to_progression_free",
        ]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                if not vals.empty:
                    censor_candidates.append(vals.max())
        censor_time = np.nan if len(censor_candidates) == 0 else max(censor_candidates)

        # choose pfs
        if prog_flag and not pd.isna(t_prog):
            pfs_time = t_prog
            pfs_event = 1
        else:
            pfs_time = censor_time
            pfs_event = 0

        return pd.Series({"PFS_time": pfs_time, "PFS_event": pfs_event})

    return (
        df.groupby(id_col)
        .apply(_compute_pfs)
        .reset_index()
        .rename(columns={id_col: "id_"})
    )

'''
def load_cptac_outcomes(
    clinical_path: Path,
    follow_up_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      outcomes[case_id] = {"PFS_time": float, "PFS_event": int, "organ": str|None}
    """
    clinical = pd.read_csv(clinical_path, sep="\t")
    follow = pd.read_csv(follow_up_path, sep="\t", na_values=["'--", "--", ""])

    pfs_df = add_pfs(follow, id_col="cases.case_id")

    if "cases.case_id" not in clinical.columns:
        raise ValueError("clinical.tsv missing required column: cases.case_id")

    clinical = clinical.copy()
    clinical["cases.case_id"] = clinical["cases.case_id"].astype(str)

    # best-effort organ columns
    organ_cols = [
        "diagnoses.tissue_or_organ_of_origin",
        "diagnoses.primary_site",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.primary_diagnosis",
    ]

    clinical_idxed = clinical.set_index("cases.case_id", drop=False)

    organ_map: Dict[str, Optional[str]] = {}
    for cid in clinical_idxed.index:
        row = clinical_idxed.loc[cid]
        organ = None
        for col in organ_cols:
            if col in row.index and pd.notna(row[col]):
                organ = str(row[col]).strip()
                break
        organ_map[str(cid)] = organ

    outcomes: Dict[str, Dict[str, Any]] = {}
    for _, r in pfs_df.iterrows():
        cid = str(r["id_"])
        t = r["PFS_time"]
        e = r["PFS_event"]
        if pd.isna(t):
            continue
        outcomes[cid] = {
            "PFS_time": float(t),
            "PFS_event": int(e),
            "organ": organ_map.get(cid, None),
        }

    return outcomes
'''

def load_cptac_outcomes(
    clinical_path: Path,
    follow_up_path: Path,
) -> Dict[str, Dict[str, Any]]:
    clinical = pd.read_csv(clinical_path, sep="\t")
    follow   = pd.read_csv(follow_up_path, sep="\t", na_values=["'--", "--", ""])

    pfs_df = add_pfs(follow, id_col="cases.case_id")

    if "cases.case_id" not in clinical.columns:
        raise ValueError("clinical.tsv missing required column: cases.case_id")

    clinical = clinical.copy()
    clinical["cases.case_id"] = clinical["cases.case_id"].astype(str)

    organ_cols = [
        "diagnoses.tissue_or_organ_of_origin",
        "diagnoses.primary_site",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.primary_diagnosis",
    ]

    clinical_idxed = clinical.drop_duplicates("cases.case_id").set_index("cases.case_id")

    organ_map: Dict[str, Optional[str]] = {}
    for cid in clinical_idxed.index:
        row = clinical_idxed.loc[cid]
        organ = None
        for col in organ_cols:
            if col in row.index and pd.notna(row[col]):
                organ = str(row[col]).strip()
                break
        organ_map[str(cid)] = organ

    outcomes: Dict[str, Dict[str, Any]] = {}
    for _, r in pfs_df.iterrows():
        cid = str(r["id_"])
        t   = r["PFS_time"]
        e   = int(r["PFS_event"])
        if pd.isna(t):
            continue

        # Incorporate death as a PFS event
        if cid in clinical_idxed.index:
            crow = clinical_idxed.loc[cid]
            vital = str(crow.get("demographic.vital_status", "")).upper()
            days_to_death = pd.to_numeric(
                crow.get("demographic.days_to_death", np.nan), errors="coerce"
            )
            if vital in ("DEAD", "DECEASED") and not pd.isna(days_to_death):
                if e == 0 or float(days_to_death) < t:
                    t = float(days_to_death)
                    e = 1

        outcomes[cid] = {
            "PFS_time":  float(t),
            "PFS_event": e,
            "organ":     organ_map.get(cid, None),
        }

    return outcomes

# -------------------------
# CPTAC mapping helpers
# -------------------------
def load_cptac_rna_filename_to_case_id(metadata_json_path: Path) -> Dict[str, str]:
    """
    Builds mapping:
      "<file_name>" -> "<case_id>"
    where file_name looks like:
      ffeda85e-....rna_seq.augmented_star_gene_counts.tsv
    """
    with open(metadata_json_path, "r") as f:
        meta = json.load(f)

    file_to_case: Dict[str, str] = {}
    for r in meta:
        ae = r.get("associated_entities", [])
        if not ae:
            continue
        case_id = ae[0].get("case_id", None)
        if case_id is None:
            continue

        for da in r.get("downstream_analyses", []):
            for of in da.get("output_files", []):
                if of.get("data_type") == "Gene Expression Quantification":
                    fn = of.get("file_name", None)
                    if fn:
                        file_to_case[fn] = case_id

    return file_to_case


def obsname_to_rna_filename(obs_name: str) -> str:
    """
    Converts AnnData obs_name into the metadata 'file_name' key.
    Example:
      obs_name = "...augmented_star_gene_counts.tsv.pkl"
      returns  "...augmented_star_gene_counts.tsv"
    """
    s = str(obs_name)
    if s.endswith(".pkl"):
        s = s[:-4]
    return s


# -------------------------
# Main preprocessing
# -------------------------
def preprocess_cptac(
    rna_seq_path: Path,
    clinical_path: Path,
    follow_up_path: Path,
    metadata_json_path: Path,
    ordered_genes_path: Path,
    gene_sets_folder: Path,
    ssgsea_folder: Path,
    emb_folder: Optional[Path],
    pca_root: Path,
    output_path: Path,
    create_FM_embeddings: bool = False,
) -> Dict:
    """
    Preprocess CPTAC data into standardized format.

    - RNA-seq input: log1p-transformed h5ad
    - Stable keys: case_id (from metadata mapping)
    - PFS_time/PFS_event: computed from clinical.tsv + follow_up.tsv
    - Organ: extracted from clinical.tsv (best-effort column matching)
    - Gene set features: max/mean pooling + ssGSEA scores
    - Embeddings: FM embeddings (optional) + PCA embeddings (always)
    """

    print("=" * 80)
    print("PREPROCESSING CPTAC DATA")
    print("=" * 80)

    print("\n[1/7] Loading RNA-seq AnnData...")
    rna_seq_info = anndata.read_h5ad(rna_seq_path)
    print(f"   Loaded RNA-seq: {rna_seq_info.shape}")

    print("\n[2/7] Loading ordered gene list...")
    ordered_genes = load_pickle(ordered_genes_path)

    print("\n[3/7] Loading CPTAC clinical + follow_up and computing PFS...")
    outcomes = load_cptac_outcomes(clinical_path, follow_up_path)
    print(f"   Cases with valid PFS_time: {len(outcomes)}")

    print("\n[4/7] Loading CPTAC metadata mapping (RNA file_name -> case_id)...")
    file_to_case = load_cptac_rna_filename_to_case_id(metadata_json_path)
    print(f"   Mapped RNA files: {len(file_to_case)}")

    print("\n[5/7] Loading ssGSEA scores...")
    ssgsea_scores = load_ssgsea_scores(ssgsea_folder, "CPTAC")

    print("\n[6/7] Processing RNA matrix + computing gene-set features...")
    rna_seq_df = rna_seq_info.to_df()
    rna_seq_df.columns = rna_seq_info.var_names.astype(str)
    rna_seq_df = rna_seq_df.groupby(rna_seq_df.columns, axis=1).max()
    rna_seq_df = ensure_genes_present(rna_seq_df, ordered_genes)
    print(f"   RNA-seq shape after ensure_genes_present: {rna_seq_df.shape}")

    gene_set_features_computed = compute_max_mean_gene_set_features(
        rna_seq_df,
        list(ordered_genes.keys()),
        gene_sets_folder,
    )
    print(f"   Computed pooled features for {len(gene_set_features_computed)} gene-set libraries")

    sample_ids = rna_seq_info.obs_names.astype(str).tolist()

    # FM embeddings (optional)
    all_embeddings = None
    if create_FM_embeddings:
        if emb_folder is None:
            raise ValueError("emb_folder must be provided when create_FM_embeddings=True")
        print("\nLoading foundation model embeddings...")
        all_embeddings = load_foundation_model_embeddings(emb_folder, sample_ids)
        for model_name, embs in all_embeddings.items():
            print(f"   {model_name}: {len(embs)} embeddings")
    else:
        print("\nSkipping foundation model embeddings (create_FM_embeddings=False)")

    # PCA embeddings (always)
    print("\n[7a/7] Loading PCA embeddings...")
    all_pca = load_pca_embeddings(pca_root, sample_ids, dims=[50, 100])
    print(f"   PCA_50:  {len(all_pca.get('PCA_50', {}))}")
    print(f"   PCA_100: {len(all_pca.get('PCA_100', {}))}")

    # -------------------------
    # Build output dict (keyed by case_id)
    # -------------------------
    print("\nBuilding output dictionary...")
    processed_data: Dict[str, Dict] = {}

    missing_case_map = 0
    missing_outcomes = 0
    missing_ssgsea = 0

    for idx, sid in enumerate(sample_ids):
        rna_filename = obsname_to_rna_filename(sid)
        case_id = file_to_case.get(rna_filename, None)
        if case_id is None:
            missing_case_map += 1
            continue

        out = outcomes.get(case_id, None)
        if out is None:
            missing_outcomes += 1
            continue

        # Gene-set features
        gs_features: Dict[str, np.ndarray] = {}

        for gs_name, feats in gene_set_features_computed.items():
            gs_features[gs_name] = feats[idx]

        for gs_name, scores_df in ssgsea_scores.items():
            if scores_df is None:
                continue
            if sid not in scores_df.index:
                missing_ssgsea += 1
                continue
            gs_features[gs_name] = scores_df.loc[sid].to_numpy()

        # Embeddings
        if create_FM_embeddings:
            embs = get_embedding_for_sample(sid, all_embeddings)
        else:
            embs = {
                "scGPT": None,
                "STATE": None,
                "CancerFoundation": None,
                "BulkFormer": None,
                "COMPASS": None,
                "BulkRNABert": None,
            }

        # Add PCA
        pca_embs = get_pca_for_sample(sid, all_pca, dims=[50, 100])
        embs.update(pca_embs)  # adds PCA_50, PCA_100

        organ = out.get("organ", None)

        processed_data[case_id] = {
            # stable identifiers
            "case_id": case_id,
            "sample_id": sid,            # original obs_name
            "rna_file": rna_filename,    # metadata 'file_name' key

            # core transcriptomic features
            "RNA_seq": rna_seq_df.iloc[idx].values,
            **gs_features,
            **embs,

            # metadata
            "dataset": "CPTAC",
            "cancer_type": organ,
            "organ": organ,

            # labels
            "PFS_time": out.get("PFS_time", None),
            "PFS_event": out.get("PFS_event", None),
        }

    print(f"   Total RNA samples: {len(sample_ids)}")
    print(f"   Missing file->case mapping: {missing_case_map}")
    print(f"   Missing outcomes for mapped cases: {missing_outcomes}")
    print(f"   Missing ssGSEA rows encountered (non-fatal): {missing_ssgsea}")
    print(f"   Saved cases: {len(processed_data)}")

    print(f"\nSaving {len(processed_data)} cases to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pickle_data(processed_data, output_path)

    print("\nCPTAC PREPROCESSING COMPLETE")
    return processed_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no_fm",
        action="store_true",
        help="If set, do not load foundation model embeddings.",
    )
    args = ap.parse_args()

    CREATE_FM_EMBEDDINGS = not args.no_fm

    REPO_ROOT = Path(__file__).resolve().parents[2]  # .../RNA-FM-benchmarking
    print(f"Repository root: {REPO_ROOT}")

    data_folder = REPO_ROOT / "data"
    cptac_folder = data_folder / "CPTAC"

    # Paths that mirror your Hartwig layout (Hartwig -> CPTAC)
    ordered_genes_path = cptac_folder / "CPTAC_genes_ordered_by_variance.pkl"
    ssgsea_folder = cptac_folder / "ssGSEA_scores"
    rna_seq_path = cptac_folder / "RNA" / "CPTAC_bulk_RNA_processed_log1p.h5ad"
    emb_folder = (cptac_folder / "RNA" / "FM_embeddings") if CREATE_FM_EMBEDDINGS else None
    pca_root = cptac_folder / "PCA"

    # CPTAC shared clinical
    clinical_path = cptac_folder / "clinical" / "clinical.tsv"
    follow_up_path = cptac_folder / "clinical" / "follow_up.tsv"

    # Metadata manifest JSON (file_name <-> case_id)
    metadata_json_path = cptac_folder / "clinical" / "metadata.cohort.all.json"

    # Gene sets
    gene_sets_folder = data_folder / "gene_sets" / "processed"

    # Output
    output_path = data_folder / "RNA_embeddings" / "CPTAC_processed.pkl"

    preprocess_cptac(
        rna_seq_path=rna_seq_path,
        clinical_path=clinical_path,
        follow_up_path=follow_up_path,
        metadata_json_path=metadata_json_path,
        ordered_genes_path=ordered_genes_path,
        gene_sets_folder=gene_sets_folder,
        ssgsea_folder=ssgsea_folder,
        emb_folder=emb_folder,
        pca_root=pca_root,
        output_path=output_path,
        create_FM_embeddings=CREATE_FM_EMBEDDINGS,
    )
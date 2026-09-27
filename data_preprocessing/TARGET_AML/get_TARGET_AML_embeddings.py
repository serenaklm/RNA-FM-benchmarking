#!/usr/bin/env python3
print("Importing...")

import argparse
import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any

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
# TARGET AML clinical helpers
# -------------------------
def obs_to_patient_id(obs_name: str) -> str:
    # TARGET aliquot IDs look like: TARGET-20-XXXXXX-09A
    # patient submitter id is:      TARGET-20-XXXXXX
    s = str(obs_name)
    parts = s.split("-")
    if len(parts) >= 4 and parts[0] == "TARGET":
        return "-".join(parts[:3])  # TARGET-20-XXXXXX
    return s


def load_target_aml_outcomes(
    aggregated_follow_up_path: Path,
    id_col: str = "cases.submitter_id",
) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(aggregated_follow_up_path)

    required = [id_col, "PFS_time", "PFS_event", "OS_time", "OS_event"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"aggregated_follow_up.csv missing required columns: {missing}")

    df = df.copy()
    df[id_col] = df[id_col].astype(str)

    def _first_valid(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else np.nan

    g = df.groupby(id_col, as_index=False).agg(
        PFS_time=("PFS_time", _first_valid),
        PFS_event=("PFS_event", _first_valid),
        OS_time=("OS_time", _first_valid),
        OS_event=("OS_event", _first_valid),
    )

    outcomes: Dict[str, Dict[str, Any]] = {}
    for _, r in g.iterrows():
        pid = str(r[id_col])

        pfs_time = pd.to_numeric(pd.Series([r["PFS_time"]]), errors="coerce").iloc[0]
        os_time = pd.to_numeric(pd.Series([r["OS_time"]]), errors="coerce").iloc[0]
        if pd.isna(pfs_time) and pd.isna(os_time):
            continue

        outcomes[pid] = {
            "PFS_time": None if pd.isna(pfs_time) else float(pfs_time),
            "PFS_event": None if pd.isna(r["PFS_event"]) else int(float(r["PFS_event"])),
            "OS_time": None if pd.isna(os_time) else float(os_time),
            "OS_event": None if pd.isna(r["OS_event"]) else int(float(r["OS_event"])),
            "organ": "Blood",
            "cancer_type": "AML",
        }

    return outcomes


# -------------------------
# Main preprocessing
# -------------------------
def preprocess_target_aml(
    rna_seq_path: Path,
    aggregated_follow_up_path: Path,
    ordered_genes_path: Path,
    gene_sets_folder: Path,
    ssgsea_folder: Path,
    emb_folder: Optional[Path],
    pca_root: Path,
    output_path: Path,
    create_FM_embeddings: bool = False,
) -> Dict:

    print("=" * 80)
    print("PREPROCESSING TARGET_AML DATA")
    print("=" * 80)

    print("\n[1/7] Loading RNA-seq AnnData...")
    rna_seq_info = anndata.read_h5ad(rna_seq_path)
    print(f"   Loaded RNA-seq: {rna_seq_info.shape}")

    print("\n[2/7] Loading ordered gene list...")
    ordered_genes = load_pickle(ordered_genes_path)

    print("\n[3/7] Loading TARGET_AML aggregated follow-up (PFS/OS)...")
    outcomes = load_target_aml_outcomes(aggregated_follow_up_path)
    print(f"   Patients with >=1 valid endpoint (PFS or OS): {len(outcomes)}")

    print("\n[4/7] Loading ssGSEA scores...")
    ssgsea_scores = load_ssgsea_scores(ssgsea_folder, "TARGET_AML")

    print("\n[5/7] Processing RNA matrix + computing gene-set features...")
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

    sample_ids_raw = rna_seq_info.obs_names.astype(str).tolist()
    patient_ids = [obs_to_patient_id(s) for s in sample_ids_raw]

    all_embeddings = None
    if create_FM_embeddings:
        if emb_folder is None:
            raise ValueError("emb_folder must be provided when create_FM_embeddings=True")
        print("\nLoading foundation model embeddings...")
        all_embeddings = load_foundation_model_embeddings(emb_folder, sample_ids_raw)
        for model_name, embs in all_embeddings.items():
            print(f"   {model_name}: {len(embs)} embeddings")
    else:
        print("\nSkipping foundation model embeddings (create_FM_embeddings=False)")

    print("\nLoading PCA embeddings...")
    all_pca = load_pca_embeddings(pca_root, sample_ids_raw, dims=[50, 100])
    print(f"   PCA_50:  {len(all_pca.get('PCA_50', {}))}")
    print(f"   PCA_100: {len(all_pca.get('PCA_100', {}))}")

    processed_data: Dict[str, Dict] = {}

    missing_outcomes = 0
    missing_ssgsea = 0

    for idx, sid in enumerate(sample_ids_raw):
        pid = patient_ids[idx]

        out = outcomes.get(pid, None)
        if out is None:
            missing_outcomes += 1
            continue

        # Gene-set pooled features from aliquot RNA row
        gs_features: Dict[str, np.ndarray] = {}
        for gs_name, feats in gene_set_features_computed.items():
            gs_features[gs_name] = feats[idx]

        # ssGSEA is indexed by aliquot/sample IDs
        for gs_name, scores_df in ssgsea_scores.items():
            if scores_df is None:
                continue
            if sid not in scores_df.index:
                missing_ssgsea += 1
                continue
            gs_features[gs_name] = scores_df.loc[sid].to_numpy()

        # FM/PCA are also aliquot/sample-level now
        if create_FM_embeddings:
            embs = get_embedding_for_sample(sid, all_embeddings)
        else:
            embs = {
                "scGPT": None,
                "STATE": None,
                "CancerFoundation": None,
                "UCE": None,
                "BulkFormer": None,
                "COMPASS": None,
                "BulkRNABert": None,
            }

        pca_embs = get_pca_for_sample(sid, all_pca, dims=[50, 100])
        embs.update(pca_embs)

        processed_data[sid] = {
            "sample_id": sid,
            "patient_id": pid,
            "case_id": pid,

            "RNA_seq": rna_seq_df.iloc[idx].values,
            **gs_features,
            **embs,

            "dataset": "TARGET_AML",
            "cancer_type": "AML",
            "organ": "Blood",

            "PFS_time": out.get("PFS_time", None),
            "PFS_event": out.get("PFS_event", None),
            "OS_time": out.get("OS_time", None),
            "OS_event": out.get("OS_event", None),
        }

    print(f"   Total RNA aliquots: {len(sample_ids_raw)}")
    print(f"   Missing outcomes (patient had no follow-up row / endpoints): {missing_outcomes}")
    print(f"   Missing ssGSEA rows encountered (non-fatal): {missing_ssgsea}")
    print(f"   Saved aliquots: {len(processed_data)}")
    print(f"   Unique patients represented: {len(set(v['patient_id'] for v in processed_data.values()))}")

    print(f"\nSaving {len(processed_data)} aliquots to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pickle_data(processed_data, output_path)

    print("\nTARGET_AML PREPROCESSING COMPLETE")
    return processed_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_fm", action="store_true", help="If set, do not load foundation model embeddings.")
    args = ap.parse_args()

    CREATE_FM_EMBEDDINGS = not args.no_fm

    REPO_ROOT = Path(__file__).resolve().parents[2]
    print(f"Repository root: {REPO_ROOT}")

    data_folder = REPO_ROOT / "data"
    target_folder = data_folder / "TARGET_AML"

    ordered_genes_path = target_folder / "TARGET_AML_genes_ordered_by_variance.pkl"
    ssgsea_folder = target_folder / "ssGSEA_scores"
    rna_seq_path = target_folder / "RNA" / "TARGET_AML_bulk_RNA_processed_log1p.h5ad"
    emb_folder = (target_folder / "RNA" / "FM_embeddings") if CREATE_FM_EMBEDDINGS else None
    pca_root = target_folder / "PCA"

    aggregated_follow_up_path = target_folder / "clinical" / "aggregated_follow_up.csv"
    gene_sets_folder = data_folder / "gene_sets" / "processed"
    output_path = data_folder / "RNA_embeddings" / "TARGET_AML_processed.pkl"

    preprocess_target_aml(
        rna_seq_path=rna_seq_path,
        aggregated_follow_up_path=aggregated_follow_up_path,
        ordered_genes_path=ordered_genes_path,
        gene_sets_folder=gene_sets_folder,
        ssgsea_folder=ssgsea_folder,
        emb_folder=emb_folder,
        pca_root=pca_root,
        output_path=output_path,
        create_FM_embeddings=CREATE_FM_EMBEDDINGS,
    )
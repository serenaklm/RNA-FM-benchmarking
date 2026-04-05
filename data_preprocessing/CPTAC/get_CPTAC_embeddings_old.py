print("Importing...")

import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from ..preprocess_utils import (
    load_foundation_model_embeddings,
    load_ssgsea_scores,
    compute_max_mean_gene_set_features,
    create_sample_dict,
    ensure_genes_present,
    get_embedding_for_sample,
    pickle_data,
    load_pickle,
)

print("Finished imports")


def preprocess_cptac(
    existing_cptac_pkl: Path,
    rna_seq_path: Path,
    gene_names: Dict[str, int],
    gene_sets_folder: Path,
    ssgsea_folder: Optional[Path],
    emb_folder: Optional[Path],
    output_path: Path,
    create_FM_embeddings: bool = False,
) -> Dict:
    """
    Preprocess CPTAC: update existing pickle with new RNA-seq, gene sets, and FM embeddings.
    Keeps existing: CNA, mutation, PFS_time, PFS_event, organ
    Adds: new RNA_seq, gene set features, ssGSEA scores, FM embeddings
    Removes: drug fields
    """
    print("=" * 80)
    print("PREPROCESSING CPTAC DATA")
    print("=" * 80)

    # -----------------------------
    # [1/7] Load existing CPTAC data
    # -----------------------------
    print("\n[1/7] Loading existing CPTAC pickle...")
    existing_data = load_pickle(existing_cptac_pkl)
    print(f"   Found {len(existing_data)} existing samples")

    # -----------------------------
    # [2/7] Load h5ad RNA-seq
    # -----------------------------
    print("\n[2/7] Loading CPTAC RNA-seq h5ad...")
    rna_seq_info = anndata.read_h5ad(rna_seq_path)
    print(f"   RNA-seq AnnData: {rna_seq_info.n_obs} x {rna_seq_info.n_vars}")

    # Normalize IDs
    rna_seq_info.obs["patient_id"] = rna_seq_info.obs["patient_id"].astype(str)
    rna_patient_ids = rna_seq_info.obs["patient_id"].tolist()
    existing_ids = list(existing_data.keys())

    overlapping = sorted(set(rna_patient_ids).intersection(set(existing_ids)))
    if len(overlapping) == 0:
        raise ValueError("No overlapping IDs between h5ad and existing pickle")

    print(f"   Found {len(overlapping)} samples with both RNA-seq and existing data")

    # Filter h5ad to overlapping IDs
    id_to_idx = {pid: i for i, pid in enumerate(rna_patient_ids)}
    keep_idx = [id_to_idx[sid] for sid in overlapping]
    rna_seq_info = rna_seq_info[keep_idx].copy()

    # -----------------------------
    # [3/7] RNA processing
    # -----------------------------
    print("\n[3/7] Processing RNA-seq data...")
    rna_seq_df = rna_seq_info.to_df()
    rna_seq_df.columns = rna_seq_info.var["gene_symbol"].astype(str).tolist()

    # Deduplicate gene symbols
    if len(set(rna_seq_df.columns)) != len(rna_seq_df.columns):
        rna_seq_df = rna_seq_df.groupby(rna_seq_df.columns, axis=1).max()

    # Reorder by CPTAC variance
    rna_seq_df = ensure_genes_present(rna_seq_df, gene_names)
    print(f"   RNA-seq shape after ensure_genes_present: {rna_seq_df.shape}")

    # Reindex to match overlapping IDs order
    rna_seq_df.index = [rna_seq_info.obs["patient_id"].iloc[i] for i in range(len(rna_seq_info))]
    rna_seq_df = rna_seq_df.loc[overlapping]

    # -----------------------------
    # [4/7] Gene set features
    # -----------------------------
    print("\n[4/7] Computing max and mean pooling gene set features...")
    gene_set_features_computed = compute_max_mean_gene_set_features(
        rna_seq_df, list(gene_names.keys()), gene_sets_folder
    )

    # -----------------------------
    # [5/7] ssGSEA scores
    # -----------------------------
    print("\n[5/7] Loading ssGSEA scores...")
    if ssgsea_folder is not None and ssgsea_folder.exists():
        ssgsea_scores = load_ssgsea_scores(ssgsea_folder, "CPTAC")
    else:
        print("   No ssGSEA folder provided/found; skipping.")
        ssgsea_scores = {
            "ssGSEA_AURORA": None,
            "ssGSEA_immuno": None,
            "ssGSEA_hallmark": None,
            "ssGSEA_combined": None,
        }

    # -----------------------------
    # [6/7] FM embeddings
    # -----------------------------
    all_embeddings = None
    if create_FM_embeddings:
        if emb_folder is None:
            raise ValueError("emb_folder must be provided when create_FM_embeddings=True")
        print("\n[6/7] Loading foundation model embeddings...")
        all_embeddings = load_foundation_model_embeddings(emb_folder, overlapping)
        for model_name, embs in all_embeddings.items():
            print(f"   {model_name}: {len(embs)} embeddings")
    else:
        print("\n[6/7] Skipping foundation model embeddings (create_FM_embeddings=False)")

    # -----------------------------
    # [7/7] Build output dictionary
    # -----------------------------
    print("\n[7/7] Building output dictionary...")
    processed_data = {}

    for idx, sid in enumerate(overlapping):
        existing_sample = existing_data[sid]

        # Gene set features for this sample
        gs_features = {k: v[idx] for k, v in gene_set_features_computed.items()}

        # ssGSEA for this sample
        for gs_name, scores_df in ssgsea_scores.items():
            if scores_df is not None and sid in scores_df.index:
                gs_features[gs_name] = scores_df.loc[sid].to_numpy()
            else:
                n_gene_sets = scores_df.shape[1] if scores_df is not None else 0
                gs_features[gs_name] = np.zeros(n_gene_sets)

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

        processed_data[sid] = create_sample_dict(
            sample_id=sid,
            rna_seq=rna_seq_df.iloc[idx].values,
            gene_set_features=gs_features,
            embeddings=embs,
            dataset="CPTAC",
            cancer_type=existing_sample.get("cancer_type", "Unknown"),
            organ=existing_sample.get("organ", "Unknown"),
            PFS_time=existing_sample.get("PFS_time"),
            PFS_event=existing_sample.get("PFS_event"),
            OS_time=None,
            OS_event=None,
            mutations=existing_sample.get("mutation"),
            cnv=existing_sample.get("CNA"),
        )

    print(f"\nSaving {len(processed_data)} samples to {output_path}")
    pickle_data(processed_data, output_path)

    print("\n" + "=" * 80)
    print("CPTAC PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total samples: {len(processed_data)}")
    if create_FM_embeddings:
        print("Foundation model embeddings: INCLUDED")
    else:
        print("Foundation model embeddings: NOT INCLUDED")

    return processed_data


if __name__ == "__main__":

    REPO_ROOT = Path(__file__).resolve().parents[2]
    print(f"Repository root: {REPO_ROOT}")

    data_folder = REPO_ROOT / "data"
    output_folder = data_folder / "RNA_embeddings"
    output_folder.mkdir(exist_ok=True, parents=True)

    CPTAC_data_folder = data_folder / "CPTAC"
    gene_names = load_pickle(CPTAC_data_folder / "shared_genes_ordered_by_CPTAC_variance.pkl")

    CREATE_FM_EMBEDDINGS = True

    processed_data = preprocess_cptac(
        existing_cptac_pkl=data_folder / "CPTAC.pkl",
        rna_seq_path=CPTAC_data_folder / "RNA" / "original_CPTAC_bulk_RNA_seq.h5ad",
        gene_names=gene_names,
        gene_sets_folder=data_folder / "gene_sets" / "processed",
        ssgsea_folder=CPTAC_data_folder / "ssGSEA_scores",
        emb_folder=CPTAC_data_folder / "RNA" / "FM_embeddings" if CREATE_FM_EMBEDDINGS else None,
        output_path=output_folder / "CPTAC_processed_v1.pkl",
        create_FM_embeddings=CREATE_FM_EMBEDDINGS,
    )
#!/usr/bin/env python3
print("Importing...")

import argparse
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
    pickle_data,
    load_pickle,
    load_pca_embeddings,
    get_pca_for_sample,
)

print("Finished imports")


def create_sample_dict_hartwig(
    sample_id: str,
    rna_seq: np.ndarray,
    gene_set_features: Dict[str, np.ndarray],
    embeddings: Dict[str, np.ndarray],
    dataset: str,
    cancer_type: Optional[str],
    organ: Optional[str],
    PFS_time: Optional[float] = None,
    PFS_event: Optional[int] = None,
    response: Optional[int] = None,
    treatment: Optional[list[str]] = None,
) -> Dict:
    """
    Create streamlined sample dictionary for Hartwig dataset.

    Only includes fields actually generated in Hartwig preprocessing.
    """

    sample = {
        # Core transcriptomic features
        "RNA_seq": rna_seq,
        **gene_set_features,
        **embeddings,

        # Metadata
        "dataset": dataset,
        "cancer_type": cancer_type,
        "organ": organ,

        # Survival / clinical labels
        "PFS_time": PFS_time,
        "PFS_event": PFS_event,
        "response": response,

        # Treatment info
        "treatment": treatment,
    }

    return sample

def _split_treatments(x: Any) -> Optional[list[str]]:
    """Parse metadata 'treatment' column where drugs are separated by '/'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "unknown"}:
        return None
    parts = [p.strip() for p in s.split("/") if p.strip()]
    return parts if parts else None


def preprocess_hartwig(
    data_key: str,
    rna_seq_path: Path,
    metadata_path: Path,
    ordered_genes_path: Path,
    gene_sets_folder: Path,
    ssgsea_folder: Path,
    labels_path: Path,
    emb_folder: Optional[Path],
    output_path: Path,
    create_FM_embeddings: bool = False,
) -> Dict:
    """
    Preprocess Hartwig data into standardized format (similar to TCGA_processed.pkl).

    - RNA-seq input: log1p-transformed h5ad
    - Labels: response/time/event dicts from hartwig_response_time_event_dicts.pkl
    - Cancer type + organ: taken directly from metadata.tsv
    - Treatments: metadata 'treatment' split by '/'
    - Gene set features: max/mean pooling + ssGSEA scores
    """

    print("=" * 80)
    print(f"PREPROCESSING {data_key} DATA")
    print("=" * 80)

    # -------------------------
    # Load core inputs
    # -------------------------
    print("\n[1/6] Loading RNA-seq AnnData...")
    rna_seq_info = anndata.read_h5ad(rna_seq_path)
    print(f"   Loaded RNA-seq: {rna_seq_info.shape} (already log1p)")

    print("\n[2/6] Loading metadata.tsv...")
    meta = pd.read_csv(metadata_path, sep="\t")
    if "sampleId" not in meta.columns:
        raise ValueError("metadata.tsv missing required column: sampleId")
    meta = meta.dropna(subset=["sampleId"]).copy()
    meta["sampleId"] = meta["sampleId"].astype(str)
    meta = meta.set_index("sampleId", drop=False)
    print(f"   Metadata rows: {len(meta)} | unique sampleIds: {meta['sampleId'].nunique()}")

    print("\n[3/6] Loading ordered gene list...")
    ordered_genes = load_pickle(ordered_genes_path)
    print(f"   Ordered genes: {len(ordered_genes)}")

    print("\n[4/6] Loading response/time/event dicts...")
    label_df = pd.read_csv(labels_path)
    label_sample_set = set(label_df["sampleId"].astype(str).unique())

    print(f"Loaded labels dataframe: {label_df.shape}")

    # Load ssGSEA scores
    print("\n[5/6] Loading ssGSEA scores...")
    ssgsea_scores = load_ssgsea_scores(ssgsea_folder, data_key)

    print("Loaded ssGSEA score tables:")
    for gs_name, df_ in ssgsea_scores.items():
        print(f"  {gs_name}: {None if df_ is None else df_.shape}")


    # -------------------------
    # Process RNA-seq matrix
    # -------------------------
    print("\n[6/6] Processing RNA-seq matrix + computing gene-set features...")
    rna_seq_df = rna_seq_info.to_df()

    # Ensure gene symbols are columns (typically var_names already)
    rna_seq_df.columns = rna_seq_info.var_names.astype(str).tolist()

    # De-duplicate gene symbols if needed
    rna_seq_df = rna_seq_df.groupby(rna_seq_df.columns, axis=1).max()

    # Ensure we have all ordered genes (and consistent order)
    rna_seq_df = ensure_genes_present(rna_seq_df, ordered_genes)
    print(f"   RNA-seq shape after ensure_genes_present: {rna_seq_df.shape}")

    # Compute pooled gene set features (max/mean)
    gene_set_features_computed = compute_max_mean_gene_set_features(
        rna_seq_df,
        list(ordered_genes.keys()),
        gene_sets_folder,
    )
    print(f"   Computed pooled features for {len(gene_set_features_computed)} gene-set libraries")
    
    # Set sample IDs
    sample_ids = rna_seq_info.obs_names.astype(str).tolist()

    # Load embeddings (optional)
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
        
    # Load PCA embeddings (always)
    pca_root = Path("/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/Hartwig/PCA") # expects PCA_50/ and PCA_100/ subfolders
    print("\nLoading PCA embeddings...")
    all_pca = load_pca_embeddings(pca_root, sample_ids, dims=[50, 100])
    print(f"   PCA_50:  {len(all_pca.get('PCA_50', {}))} embeddings")
    print(f"   PCA_100: {len(all_pca.get('PCA_100', {}))} embeddings")

    # -------------------------
    # Build output dict
    # -------------------------
    print("\nBuilding output dictionary...")
    processed_data: Dict[str, Dict] = {}

    sample_ids = rna_seq_info.obs_names.astype(str).tolist()

    missing_meta = 0
    missing_labels = 0
    
    num_filtered_out = 0

    for idx, sid in enumerate(sample_ids):
        # metadata
        if sid not in meta.index:
            missing_meta += 1
            continue

        m = meta.loc[sid]

        cancer_type = m.get("primaryTumorType", None)
        organ = m.get("primaryTumorLocation", None)

        # treatments from metadata.tsv 'treatment' field
        treatment_list = _split_treatments(m.get("treatment", None))

        # labels
        if sid in label_sample_set:
            row = label_df.loc[label_df["sampleId"] == sid]
            resp = row["response"].values[0]
            t_time = row["time"].values[0]
            t_event = row["event"].values[0]
        else:
            resp, t_time, t_event = None, None, None

        # We can include samples even without labels; but for survival tasks you may filter later.
        # Here we keep them, with None labels.
        if (t_time is None) or (t_event is None):
            missing_labels += 1

        # gene-set features
        gs_features: Dict[str, np.ndarray] = {}

        # pooled (max/mean) gene-set features
        for gs_name, feats in gene_set_features_computed.items():
            gs_features[gs_name] = feats[idx]

        # Get ssGSEA scores for this sample (Hartwig obs_names are already sampleIds)
        for gs_name, scores_df in ssgsea_scores.items():
            gs_features[gs_name] = scores_df.loc[sid].to_numpy()

        # embeddings
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
        
        # Add PCA embeddings (always)
        pca_embs = get_pca_for_sample(sid, all_pca, dims=[50, 100])
        embs.update(pca_embs)  # adds keys: PCA_50, PCA_100

        sample_dict = create_sample_dict_hartwig(
            sample_id=sid,
            rna_seq=rna_seq_df.iloc[idx].values,
            gene_set_features=gs_features,
            embeddings=embs,
            dataset=data_key,
            cancer_type=cancer_type,
            organ=organ,
            PFS_time=float(t_time) if t_time is not None else None,
            PFS_event=int(t_event) if t_event is not None else None,
            response=int(resp) if resp is not None else None,
            treatment=treatment_list,
        )

        # Attach response as an extra field (not part of create_sample_dict signature)
        sample_dict["response"] = int(resp) if resp is not None and not pd.isna(resp) else None
        
        if sid in label_sample_set:
            processed_data[sid] = sample_dict
        else:
            num_filtered_out += 1

    print(f"   Total samples in RNA: {len(sample_ids)}")
    print(f"   Samples missing metadata: {missing_meta}")
    print(f"   Samples missing time/event labels (kept with None): {missing_labels}")
    print(f"   Samples saved: {len(processed_data)}")
    print(f"   Filtered {num_filtered_out} patients")

    # Save
    print(f"\nSaving {len(processed_data)} samples to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pickle_data(processed_data, output_path)

    print("\n" + "=" * 80)
    print(f"{data_key} PREPROCESSING COMPLETE")
    print("=" * 80)

    return processed_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_key", type=str, required=True, choices=["Hartwig", "Hartwig_500"])
    args = ap.parse_args()

    data_key = args.data_key
    CREATE_FM_EMBEDDINGS = True # Set this

    REPO_ROOT = Path(__file__).resolve().parents[2]  # .../RNA-FM-benchmarking
    print(f"Repository root: {REPO_ROOT}")

    data_folder = REPO_ROOT / "data"
    hartwig_folder = data_folder / data_key
    output_folder = data_folder / "RNA_embeddings"
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Dataset-specific folders
    ds_folder = data_folder / data_key
    ordered_genes_path = hartwig_folder / f"{data_key}_genes_ordered_by_variance.pkl"
    ssgsea_folder = hartwig_folder / "ssGSEA_scores"
    
    

    if data_key == "Hartwig_500":
        rna_seq_path = hartwig_folder / "RNA" / "Hartwig_500_bulk_RNA_processed_log1p.h5ad"

    else:
        rna_seq_path = hartwig_folder / "RNA" / "Hartwig_bulk_RNA_processed_log1p.h5ad"

    # Can be subsetted if only looking at 500 patients later
    labels_path = hartwig_folder / "clinical_labels.csv"
    metadata_path = Path("/data/rbg/shared/datasets/Hartwig/isofox_tpm_update_4/metadata.tsv")

    # Gene sets folder
    gene_sets_folder = data_folder / "gene_sets" / "processed"

    # Optional FM embeddings folder (only used if --create_FM_embeddings)
    emb_folder = (hartwig_folder / "RNA" / "FM_embeddings") if CREATE_FM_EMBEDDINGS else None

    preprocess_hartwig(
        data_key=data_key,
        rna_seq_path=rna_seq_path,
        metadata_path=metadata_path,
        ordered_genes_path=ordered_genes_path,
        gene_sets_folder=gene_sets_folder,
        ssgsea_folder=ssgsea_folder,
        labels_path=labels_path,
        emb_folder=emb_folder,
        output_path=output_folder / f"{data_key}_processed.pkl",
        create_FM_embeddings=CREATE_FM_EMBEDDINGS,
    )
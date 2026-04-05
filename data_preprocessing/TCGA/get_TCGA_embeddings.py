print("Importing...")

import anndata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Import from data_preprocessing package
from ..preprocess_utils import (
    load_foundation_model_embeddings,
    load_ssgsea_scores,
    compute_max_mean_gene_set_features,
    create_sample_dict,
    ensure_genes_present,
    get_embedding_for_sample,
    load_json, 
    pickle_data,
    load_pickle,
    build_vocab_from_lists,
    one_hot_from_vocab,
    load_pca_embeddings,
    get_pca_for_sample,
)
print("Finished imports")

ORGAN_MAPPING = {
    "TCGA-ACC":  "Adrenal gland",
    "TCGA-BLCA": "Bladder",
    "TCGA-BRCA": "Breast",
    "TCGA-CESC": "Cervix",
    "TCGA-CHOL": "Bile duct",
    "TCGA-COAD": "Colon",
    "TCGA-DLBC": "Lymph node",
    "TCGA-ESCA": "Esophagus",
    "TCGA-GBM":  "Brain",
    "TCGA-HNSC": "Head and neck",
    "TCGA-KICH": "Kidney",
    "TCGA-KIRC": "Kidney",
    "TCGA-KIRP": "Kidney",
    "TCGA-LAML": "Blood",
    "TCGA-LGG":  "Brain",
    "TCGA-LIHC": "Liver",
    "TCGA-LUAD": "Lung",
    "TCGA-LUSC": "Lung",
    "TCGA-MESO": "Pleura",
    "TCGA-OV":   "Ovary",
    "TCGA-PAAD": "Pancreas",
    "TCGA-PCPG": "Adrenal gland",
    "TCGA-PRAD": "Prostate",
    "TCGA-READ": "Colon",
    "TCGA-SARC": "Soft tissue",
    "TCGA-SKCM": "Skin",
    "TCGA-STAD": "Stomach",
    "TCGA-TGCT": "Testis",
    "TCGA-THCA": "Thyroid",
    "TCGA-THYM": "Thymus",
    "TCGA-UCEC": "Uterus",
    "TCGA-UCS":  "Uterus",
    "TCGA-UVM":  "Eye"
}

def compute_OS(clinical_features: pd.DataFrame) -> pd.DataFrame:
    """Compute overall survival time and event from clinical features."""
    df = clinical_features.copy()
    
    # Vital status → event (1 = death, 0 = censored, else NaN)
    vital = (
        df['demographic.vital_status'].astype(str).str.lower()
        if 'demographic.vital_status' in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    os_event = np.where(vital == 'dead', 1,
                np.where(vital == 'alive', 0, np.nan)).astype(float)
    
    # Death time (used only if event == 1)
    death_time = (
        pd.to_numeric(df['demographic.days_to_death'], errors='coerce')
        if 'demographic.days_to_death' in df.columns
        else pd.Series(np.nan, index=df.index, dtype='float')
    )
    
    # Last-contact time fallbacks
    last_contact_candidates = [
        'diagnoses.days_to_last_follow_up',
        'diagnoses.days_to_last_followup',
        'cases.days_to_last_known_alive',
        'cases.days_to_last_follow_up',
        'cases.days_to_last_followup',
        'cases.days_to_lost_to_followup',
    ]
    last_contact = pd.Series(np.nan, index=df.index, dtype='float')
    for c in last_contact_candidates:
        if c in df.columns:
            last_contact = last_contact.combine_first(pd.to_numeric(df[c], errors='coerce'))
    
    # OS time: if event==1 use death_time; else use last_contact
    os_time = np.where(os_event == 1, death_time, last_contact)
    os_time = pd.to_numeric(os_time, errors='coerce')
    os_time = np.where(os_time < 0, np.nan, os_time)
    
    df['OS_time'] = os_time
    df['OS_event'] = os_event
    
    return df

def compute_PFS(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Compute progression-free survival time and event."""
    outcomes = outcomes.copy()
    
    outcomes.loc[:, "PFS_time"] = outcomes.apply(
        lambda row: row["days_to_progression_recurrence"] if row["progression_recurrence_event"] == 1
        else row["max_follow_up_days"],
        axis=1
    )
    
    outcomes["PFS_event"] = ((outcomes["progression_recurrence_event"] == 1)).astype(int)
    
    return outcomes

def get_patients_treatment(clinical_features: pd.DataFrame, 
                          drug_moa_mapping: Dict[str, str]) -> Dict[str, Dict]:
    """
    Extract treatment information for each patient.
    
    Returns:
        Dict mapping patient_id to {"treatment": [drugs], "drug_moa": [moa_categories]}
    """
    treatments = {}
    
    for patient_id, data in clinical_features.groupby("cases.submitter_id"):
        treatment_list = set(data["treatments.therapeutic_agents"].unique())
        treatment_list = list(set([
            d for d in treatment_list 
            if d not in ["'--", "Unknown", "Clinical Trial", "Not Reported", "Clinical Trial Agent"]
        ]))
        
        if len(treatment_list) == 0:
            continue
            
        # Get mechanism of action
        treatment_moa = []
        for drug in treatment_list:
            if drug in drug_moa_mapping and drug_moa_mapping[drug] != "unclear":
                treatment_moa.append(drug_moa_mapping[drug])
        
        treatments[patient_id] = {
            "treatment": treatment_list,
            "drug_moa": list(set(treatment_moa)) if treatment_moa else None
        }
    
    return treatments

def preprocess_tcga(
    rna_seq_path: Path,
    clinical_features_path: Path,
    outcomes_path: Path,
    drug_moa_path: Path,
    gene_names: Dict[str, int],
    gene_sets_folder: Path,
    ssgsea_folder: Path,
    emb_folder: Optional[Path],
    output_path: Path,
    create_FM_embeddings: bool = False
) -> Dict:
    """
    Preprocess TCGA data into standardized format.
    
    Args:
        rna_seq_path: Path to RNA-seq h5ad file (already log1p transformed)
        clinical_features_path: Path to clinical features TSV
        outcomes_path: Path to outcomes CSV
        drug_moa_path: Path to drug mechanism of action JSON
        gene_names: Dictionary of gene names
        gene_sets_folder: Path to folder containing gene set definitions
        ssgsea_folder: Path to folder containing ssGSEA score pickle files
        emb_folder: Path to folder containing foundation model embeddings (optional)
        output_path: Path to save processed data
        create_FM_embeddings: Whether to load and include foundation model embeddings
    
    Returns:
        Dict mapping patient_id to sample dictionary
    """
    
    print("=" * 80)
    print("PREPROCESSING TCGA DATA")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading TCGA data...")
    rna_seq_info = anndata.read_h5ad(rna_seq_path)
    print(f"   RNA-seq data already log1p transformed")
    
    clinical_features = pd.read_csv(clinical_features_path, sep="\t")
    outcomes = pd.read_csv(outcomes_path)
    drug_moa_mapping = {r["Drug"]: r["Mechanism"] for r in load_json(drug_moa_path)}
    
    # Process patient IDs
    rna_seq_info.obs["cases.submitter_id"] = rna_seq_info.obs["cases.submitter_id"].apply(lambda x: x[:12])
    patient_id_list = rna_seq_info.obs["cases.submitter_id"].values
    
    # Filter to patients with outcomes
    patient_id_list_w_outcomes = list(outcomes["patient_id"].values)
    overlapping_patients = list(set(patient_id_list).intersection(set(patient_id_list_w_outcomes)))
    rna_seq_info = rna_seq_info[rna_seq_info.obs["cases.submitter_id"].isin(overlapping_patients)]
    print(f"   Found {len(overlapping_patients)} patients with both RNA-seq and outcomes")
    
    # Get uuids
    sample_uuid_list = rna_seq_info.obs_names.astype(str).tolist()
    
    # Compute survival outcomes
    print("\n[2/7] Computing survival outcomes...")
    outcomes = compute_PFS(outcomes)
    clinical_features = compute_OS(clinical_features)
    
    # Merge OS into outcomes
    clinical_features_subset = clinical_features[["cases.submitter_id", "OS_time", "OS_event"]].copy()
    clinical_features_subset.columns = ["patient_id", "OS_time", "OS_event"]
    outcomes = outcomes.merge(clinical_features_subset, on="patient_id", how="left")
    
    # Process RNA-seq
    print("\n[3/7] Processing RNA-seq data...")
    rna_seq_df = rna_seq_info.to_df()
    rna_seq_df.columns = rna_seq_info.var["gene_symbol"].tolist()
    rna_seq_df = rna_seq_df.groupby(rna_seq_df.columns, axis=1).max()
    rna_seq_df = ensure_genes_present(rna_seq_df, gene_names)
    print(f"   RNA-seq shape: {rna_seq_df.shape}")
    
    # Compute max and mean gene set features
    print("\n[4/7] Computing max and mean pooling gene set features...")
    gene_set_features_computed = compute_max_mean_gene_set_features(
        rna_seq_df, list(gene_names.keys()), gene_sets_folder
    )
    
    # Load ssGSEA scores
    print("\n[5/7] Loading ssGSEA scores...")
    ssgsea_scores = load_ssgsea_scores(ssgsea_folder, "TCGA")
    
    # Load embeddings (conditionally)
    all_embeddings = None
    if create_FM_embeddings:
        if emb_folder is None:
            raise ValueError("emb_folder must be provided when create_FM_embeddings=True")
        print("\n[6/7] Loading foundation model embeddings...")
        patient_ids = rna_seq_info.obs["cases.submitter_id"].values.tolist()
        all_embeddings = load_foundation_model_embeddings(emb_folder, patient_ids)
        for model_name, embs in all_embeddings.items():
            print(f"   {model_name}: {len(embs)} embeddings")
    else:
        print("\n[6/7] Skipping foundation model embeddings (create_FM_embeddings=False)")
    
    # PCA root for this dataset
    pca_root = Path(f"/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/TCGA/PCA/")
    print("\n[6a/7] Loading PCA embeddings...")
    all_pca = load_pca_embeddings(pca_root, patient_ids, dims=[50, 100])
    print(f"   PCA_50:  {len(all_pca.get('PCA_50', {}))} embeddings")
    print(f"   PCA_100: {len(all_pca.get('PCA_100', {}))} embeddings")
    
    # Get treatment information
    print("\n[7/7] Extracting treatment information...")
    treatments = get_patients_treatment(clinical_features, drug_moa_mapping)
    print(f"   Found treatment info for {len(treatments)} patients")
    
    # Build MoA vocabulary across all patients (for consistent one-hot encoding)
    all_moa_lists = [v.get("drug_moa", None) for v in treatments.values()]
    drug_moa_vocab = build_vocab_from_lists(all_moa_lists)
    print(f"   Unique drug MoAs (vocab size): {len(drug_moa_vocab)}")

    # Save vocab for downstream reference
    moa_vocab_path = "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/TCGA/TCGA_drug_moa_vocab.pkl"
    pickle_data(drug_moa_vocab, moa_vocab_path)
    print(f"   Saved drug MoA vocab to {moa_vocab_path}")
    
    # Build output dictionary
    print("\nBuilding output dictionary...")
    processed_data = {}
    patient_ids = rna_seq_info.obs["cases.submitter_id"].values.tolist()
    
    for idx, patient_id in enumerate(patient_ids):
        # Get outcomes for this patient
        patient_outcomes = outcomes[outcomes["patient_id"] == patient_id]
        if patient_outcomes.shape[0] == 0:
            continue
        patient_outcomes = patient_outcomes.iloc[0]
        
        # Get cancer type and organ
        cancer_type = rna_seq_info.obs.iloc[idx]["project.project_id"]
        
        organ = ORGAN_MAPPING.get(
            cancer_type,
            "Unknown"
        )
        
        cancer_type = cancer_type.removeprefix("TCGA-")
        
        # Get max/mean gene set features for this patient
        gs_features = {}
        for gs_name, feats in gene_set_features_computed.items():
            gs_features[gs_name] = feats[idx]
        
        # Get ssGSEA scores for this patient
        sample_uuid = sample_uuid_list[idx]

        for gs_name, scores_df in ssgsea_scores.items():
            if scores_df is not None and sample_uuid in scores_df.index:
                gs_features[gs_name] = scores_df.loc[sample_uuid].to_numpy()
            else:
                n_gene_sets = scores_df.shape[1] if scores_df is not None else 0
                gs_features[gs_name] = np.zeros(n_gene_sets)
        
        # Get embeddings for this patient (conditionally)
        if create_FM_embeddings:
            embs = get_embedding_for_sample(patient_id, all_embeddings)
        else:
            # Provide None values for all embeddings
            embs = {
                "scGPT": None,
                "STATE": None,
                "CancerFoundation": None,
                "BulkFormer": None,
                "COMPASS": None,
                "BulkRNABert": None
            }
        
        # Add PCA embeddings (always)
        pca_embs = get_pca_for_sample(patient_id, all_pca, dims=[50, 100])
        embs.update(pca_embs)   # adds keys: PCA_50, PCA_100
        
        # Treatment info (if available)
        treatment_info = treatments.get(patient_id, {"treatment": None, "drug_moa": None})
        
        # Compute one-hot encoding for drug MoA
        drug_moa_one_hot = one_hot_from_vocab(treatment_info["drug_moa"], drug_moa_vocab)
        
        # Create standardized sample
        processed_data[patient_id] = create_sample_dict(
            sample_id=patient_id,
            rna_seq=rna_seq_df.iloc[idx].values,
            gene_set_features=gs_features,
            embeddings=embs,
            dataset="TCGA",
            cancer_type=cancer_type,
            organ=organ,
            PFS_time=float(patient_outcomes["PFS_time"]) if not pd.isna(patient_outcomes["PFS_time"]) else None,
            PFS_event=int(patient_outcomes["PFS_event"]) if not pd.isna(patient_outcomes["PFS_event"]) else None,
            OS_time=float(patient_outcomes["OS_time"]) if not pd.isna(patient_outcomes["OS_time"]) else None,
            OS_event=int(patient_outcomes["OS_event"]) if not pd.isna(patient_outcomes["OS_event"]) else None,
            treatment=treatment_info["treatment"],
            drug_moa=treatment_info["drug_moa"],
            drug_moa_one_hot=drug_moa_one_hot
        )
    
    # Save
    print(f"\nSaving {len(processed_data)} patients to {output_path}")
    pickle_data(processed_data, output_path)
    
    print("\n" + "=" * 80)
    print("TCGA PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total patients: {len(processed_data)}")
    print(f"Patients with treatment info: {sum(1 for p in processed_data.values() if p['treatment'] is not None)}")
    print(f"Patients with drug MoA info: {sum(1 for p in processed_data.values() if p['drug_moa'] is not None)}")
    if create_FM_embeddings:
        print(f"Foundation model embeddings: INCLUDED")
    else:
        print(f"Foundation model embeddings: NOT INCLUDED")
    
    return processed_data

if __name__ == "__main__":
    
    REPO_ROOT = Path(__file__).resolve().parents[2]   # .../RNA-FM-benchmarking
    print(f"Repository root: {REPO_ROOT}")
    
    data_folder  = REPO_ROOT / "data"
    output_folder = data_folder / "RNA_embeddings"
    output_folder.mkdir(exist_ok=True, parents=True)
    
    TCGA_data_folder = data_folder / "TCGA"
    gene_names = load_pickle(TCGA_data_folder / "TCGA_genes_ordered_by_variance.pkl")
    
    CREATE_FM_EMBEDDINGS = True
    
    processed_data = preprocess_tcga(
        rna_seq_path=TCGA_data_folder / "RNA" / "TCGA_bulk_RNA_processed_log1p.h5ad",
        clinical_features_path=TCGA_data_folder / "clinical" / "clinical.tsv",
        outcomes_path=TCGA_data_folder / "clinical" / "clinical_data_prog_rec_death.csv",
        drug_moa_path=TCGA_data_folder / "TCGA_drug_moa.json",
        gene_names=gene_names,
        gene_sets_folder=data_folder / "gene_sets" / "processed",
        ssgsea_folder=TCGA_data_folder / "ssGSEA_scores",
        emb_folder=TCGA_data_folder / "RNA" / "FM_embeddings" if CREATE_FM_EMBEDDINGS else None,
        output_path=output_folder / "TCGA_processed.pkl",
        create_FM_embeddings=CREATE_FM_EMBEDDINGS
    )
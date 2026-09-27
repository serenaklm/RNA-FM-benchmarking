import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import pickle
from typing import Iterable

def build_vocab_from_lists(list_of_lists: Iterable[Optional[List[str]]]) -> List[str]:
    """Flatten lists, drop None/empty, and return a sorted unique vocab."""
    s = set()
    for x in list_of_lists:
        if x is None:
            continue
        for v in x:
            if v is None:
                continue
            s.add(str(v))
    return sorted(s)

def one_hot_from_vocab(items: Optional[List[str]], vocab: List[str]) -> np.ndarray:
    """Return (len(vocab),) float32 one-hot vector for 'items'."""
    if items is None:
        return np.zeros(len(vocab), dtype=np.float32)
    item_set = set(items)
    return np.array([1.0 if v in item_set else 0.0 for v in vocab], dtype=np.float32)

def pickle_data(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path): 

    with open(path, "rb") as f: 
        return pickle.load(f)

def load_json(path):

    with open(path, "r", encoding = "UTF-8") as f: 
        return json.load(f)

def write_txt(data, path):

    with open(path, "w", encoding = "UTF-8") as f:

        for l in data:
            f.write(f"{l}\n")

def load_emb(path: Path) -> np.ndarray:
    """Load embedding from pickle file and convert to numpy array."""
    emb = load_pickle(path)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    elif not isinstance(emb, np.ndarray):
        emb = np.asarray(emb)
    return emb

def load_foundation_model_embeddings(emb_folder: Path, sample_ids: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all foundation model embeddings for given sample IDs.
    
    Args:
        emb_folder: Folder containing subdirectories for each model
        sample_ids: List of sample IDs to load
        
    Returns:
        Dict mapping model_name -> {sample_id -> embedding as numpy array}
    """
    model_names = ["scGPT", "STATE", "CancerFoundation", "UCE", "BulkFormer", "BulkRNABert", "COMPASS"]
    embeddings = {name: {} for name in model_names}
    
    for model_name in model_names:
        model_folder = emb_folder / model_name
        if not model_folder.exists():
            print(f"Warning: {model_name} folder not found at {model_folder}")
            continue
            
        for f in os.listdir(model_folder):
            if not f.endswith(".pkl"):
                continue
            sample_id = f.replace(".pkl", "")
            if sample_id in sample_ids:
                embedding = load_emb(model_folder / f)
                
                # Convert to numpy array
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu().numpy()
                elif isinstance(embedding, np.ndarray):
                    pass  # already numpy
                else:
                    embedding = np.array(embedding)
                
                embeddings[model_name][sample_id] = embedding
                
        print(f"Loaded {len(embeddings[model_name])} embeddings for {model_name}")
    
    return embeddings

def get_gene_set_features(RNA_seq: pd.DataFrame, gene_names: List[str], 
                          gene_sets: Dict[str, List[str]], 
                          aggregation: str = "max") -> np.ndarray:
    """
    Extract gene set features using specified aggregation method.
    
    Args:
        RNA_seq: DataFrame with gene expression (samples x genes)
        gene_names: List of gene names (columns)
        gene_sets: Dict mapping gene set names to lists of genes
        aggregation: "max" or "mean"
        
    Returns:
        Array of shape (n_samples, n_gene_sets)
    """
    RNA_seq.columns = gene_names
    gs_features = []
    
    for gs_name, genes in gene_sets.items():
        # Remove genes not in columns
        genes = [gn for gn in genes if gn in gene_names]
        if len(genes) == 0:
            continue
            
        if aggregation == "max":
            f = RNA_seq[genes].max(axis=1)
        elif aggregation == "mean":
            f = RNA_seq[genes].mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Supported: 'max', 'mean'")
            
        gs_features.append(f)
    
    if len(gs_features) == 0:
        return np.zeros((len(RNA_seq), 0))
        
    gs_features = pd.concat(gs_features, axis=1).values
    return gs_features

def compute_max_mean_gene_set_features(
    rna_seq_df: pd.DataFrame,
    gene_names: list,
    gene_sets_folder: Path
) -> Dict[str, np.ndarray]:
    """
    Compute max and mean pooling gene set features.
    
    Args:
        rna_seq_df: DataFrame with gene expression (samples x genes)
        gene_names: List of gene names
        gene_sets_folder: Path to folder containing gene set definitions
    
    Returns:
        Dict with keys like "max_AURORA", "mean_AURORA", "max_immuno", etc.
    """
    print(f"   Computing max and mean pooling gene set features...")
    
    gene_sets_dict = load_gene_sets(gene_sets_folder)
    gene_set_features = {}
    
    # Compute for max and mean aggregations
    for agg in ["max", "mean"]:
        # Process each gene set collection (AURORA, immuno, hallmark, combined)
        for gs_name, gs_genes in gene_sets_dict.items():
            key = f"{agg}_{gs_name}"
            gene_set_features[key] = get_gene_set_features(
                rna_seq_df, gene_names, gs_genes, agg
            )
            print(f"      {key}: {gene_set_features[key].shape}")
    
    return gene_set_features

def load_gene_sets(gene_sets_folder: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load all gene set collections."""
    gene_sets = {}
    
    gene_set_files = {
        "AURORA": "AURORA_gene_sets.pkl",
        "immuno": "immuno_gene_sets.pkl",
        "hallmark": "50_hallmark_gene_sets.pkl",
        "combined": "combined_gene_sets.pkl"
    }
    
    for name, filename in gene_set_files.items():
        path = gene_sets_folder / filename
        if path.exists():
            gene_sets[name] = load_pickle(path)
        else:
            print(f"Warning: Gene set file not found: {path}")
            gene_sets[name] = {}
    
    return gene_sets

def compute_all_gene_set_features(RNA_seq_df: pd.DataFrame, 
                                   gene_names: List[str],
                                   gene_sets_folder: Path,
                                   aggregation: str = "max") -> Dict[str, np.ndarray]:
    """
    Compute gene set features for all gene set collections.
    
    Returns:
        Dict mapping gene_set_name -> feature array
    """
    gene_sets_dict = load_gene_sets(gene_sets_folder)
    
    gene_set_features = {}
    
    # Individual gene set collections
    for gs_name, gs_genes in gene_sets_dict.items():
        gene_set_features[f"{gs_name}_gs"] = get_gene_set_features(
            RNA_seq_df, gene_names, gs_genes, aggregation
        )
    
    # Combined gene sets
    all_gene_sets = {}
    for gs_dict in gene_sets_dict.values():
        all_gene_sets.update(gs_dict)
    
    gene_set_features["all_gs"] = get_gene_set_features(
        RNA_seq_df, gene_names, all_gene_sets, aggregation
    )
    
    return gene_set_features

def create_sample_dict(
    sample_id: str,
    rna_seq: np.ndarray,
    gene_set_features: Dict[str, np.ndarray],
    embeddings: Dict[str, np.ndarray],
    dataset: str,
    cancer_type: Optional[str] = None,
    organ: Optional[str] = None,
    **task_specific_fields
) -> Dict:
    """
    Create standardized sample dictionary.
    
    Args:
        sample_id: Sample identifier
        rna_seq: Raw RNA-seq array
        gene_set_features: Dict of gene set features (e.g., {"AURORA_gs": array, ...})
        embeddings: Dict of foundation model embeddings
        dataset: Dataset name (TCGA, GDSC, etc.)
        cancer_type: Cancer type or cell line lineage
        **task_specific_fields: Additional fields (PFS_time, AUC, etc.)
    """
    sample = {
        # Core features
        "RNA_seq": rna_seq,
        **gene_set_features,
        **embeddings,
        
        # Metadata
        "dataset": dataset,
        "cancer_type": cancer_type,
        "organ": organ,
        
        # Initialize all possible task-specific fields as None
        "PFS_time": None,
        "PFS_event": None,
        "OS_time": None,
        "OS_event": None,
        "drug": None,
        "IC50": None,
        "AUC": None,
        "cell_line": None,
        "ici_response": None,
        "response_type": None,
        "drug_moa": None,
        "drug_moa_one_hot": None,
        "treatment": None,
        "mutations": None,
        "cnv": None,
    }
    
    # Update with provided task-specific fields
    sample.update(task_specific_fields)
    
    return sample

def ensure_genes_present(rna_seq_df: pd.DataFrame, gene_names: Dict[str, int]) -> pd.DataFrame:
    """
    Ensure all required genes are present in RNA-seq dataframe.
    Add missing genes with zero values.
    """
    for gene in gene_names.keys():
        if gene not in rna_seq_df.columns:
            rna_seq_df[gene] = 0
    
    # Order genes by decreasing variability
    return rna_seq_df.loc[:, list(gene_names.keys())]

def get_embedding_for_sample(sample_id: str, 
                             all_embeddings: Dict[str, Dict[str, np.ndarray]],
                             default_dim: int = 512) -> Dict[str, np.ndarray]:
    """
    Get embeddings for a single sample, with default values if not found.
    
    Returns:
        Dict mapping model_name -> embedding array
    """
    return {
        "scGPT": all_embeddings["scGPT"].get(sample_id, np.zeros(default_dim)),
        "STATE": all_embeddings["STATE"].get(sample_id, np.zeros(default_dim)),
        "CancerFoundation": all_embeddings["CancerFoundation"].get(sample_id, np.zeros(default_dim)),
        "UCE": all_embeddings["UCE"].get(sample_id, np.zeros(default_dim)),
        "BulkFormer": all_embeddings["BulkFormer"].get(sample_id, np.zeros(default_dim)),
        "BulkRNABert": all_embeddings["BulkRNABert"].get(sample_id, np.zeros(default_dim)),
        "COMPASS": all_embeddings["COMPASS"].get(sample_id, np.zeros(default_dim))
    }
    
def load_ssgsea_scores(ssgsea_folder: Path, dataset_name: str = "TCGA") -> Dict[str, pd.DataFrame]:
    """
    Load pre-computed ssGSEA score dataframes.
    
    Args:
        ssgsea_folder: Path to folder containing ssGSEA pickle files
        dataset_name: Name of dataset (for logging)
    
    Returns:
        Dict mapping "ssGSEA_{gene_set_name}" to DataFrame (index=sample_id, columns=gene_sets)
    """
    print(f"   Loading ssGSEA scores from {ssgsea_folder}...")
    
    ssgsea_files = {
        "ssGSEA_AURORA": "aurora.pkl",
        "ssGSEA_immuno": "immuno.pkl",
        "ssGSEA_hallmark": "hallmark.pkl",
        "ssGSEA_combined": "combined.pkl"
    }
    
    ssgsea_scores = {}
    for gs_name, filename in ssgsea_files.items():
        filepath = ssgsea_folder / filename
        if filepath.exists():
            df = load_pickle(filepath)
            ssgsea_scores[gs_name] = df
            print(f"      {gs_name}: {df.shape} (samples x gene sets)")
        else:
            print(f"      Warning: {filepath} not found, skipping {gs_name}")
            ssgsea_scores[gs_name] = None
    
    return ssgsea_scores

def load_pca_embeddings(
    pca_root: Path,
    sample_ids: List[str],
    dims: List[int] = [50, 100],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load PCA embeddings stored as per-sample pickle files:
      {pca_root}/PCA_50/{sample_id}.pkl
      {pca_root}/PCA_100/{sample_id}.pkl

    Returns:
      {
        "PCA_50": {sample_id: np.ndarray(50,)},
        "PCA_100": {sample_id: np.ndarray(100,)}
      }
    """
    out: Dict[str, Dict[str, np.ndarray]] = {f"PCA_{d}": {} for d in dims}

    for d in dims:
        key = f"PCA_{d}"
        folder = pca_root / key
        if not folder.exists():
            print(f"Warning: PCA folder not found: {folder}")
            continue

        for sid in sample_ids:
            fp = folder / f"{sid}.pkl"
            if not fp.exists():
                continue
            out[key][sid] = load_emb(fp).astype(np.float32)

    return out


def get_pca_for_sample(
    sample_id: str,
    all_pca: Optional[Dict[str, Dict[str, np.ndarray]]],
    dims: List[int] = [50, 100],
) -> Dict[str, np.ndarray]:
    """
    Return PCA embeddings for a sample, with zeros if missing.
    """
    out = {}
    for d in dims:
        key = f"PCA_{d}"
        default = np.zeros(d, dtype=np.float32)
        if all_pca is None:
            out[key] = default
        else:
            out[key] = all_pca.get(key, {}).get(sample_id, default)
    return out


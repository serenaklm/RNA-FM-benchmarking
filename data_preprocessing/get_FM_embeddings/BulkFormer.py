"""
BulkFormer embeddings. Copied into this repo from the BulkFormer checkout
(BULKFORMER_DIR/get_BulkFormer_embeddings.py) so the
benchmark's embedding code is versioned here. Runs with the BulkFormer repo as working
directory, since the model code and weights are loaded relative to it.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BULKFORMER_DIR = os.environ.get("BULKFORMER_DIR", "/data/rbg/users/ashyamal/BulkFormer/BulkFormer")
_INVOCATION_CWD = os.getcwd()
os.chdir(BULKFORMER_DIR)
sys.path.insert(0, BULKFORMER_DIR)
import os
import math
import pickle
import mygene
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,random_split
from torch_geometric.typing import SparseTensor
from utils.BulkFormer import BulkFormer
from model.config import model_params
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_common import add_subset_args, restrict_to_samples, sample_ids_of

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, required=True,
                help='Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...)')
parser.add_argument('--just_preprocess_data', action='store_true')
parser.add_argument(
    '--data_dir', type=str, default=None,
    help='Override data directory. If set, reads {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad '
         'and writes to {data_dir}/RNA/FM_embeddings/BulkFormer/.',
)
add_subset_args(parser)
args = parser.parse_args()
if args.data_dir and not os.path.isabs(args.data_dir):
    args.data_dir = os.path.join(_INVOCATION_CWD, args.data_dir)
if args.sample_ids and not os.path.isabs(args.sample_ids):
    args.sample_ids = os.path.join(_INVOCATION_CWD, args.sample_ids)

if args.data_dir:
    data_path = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
    output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "BulkFormer") + "/"
else:
    data_path = f"{REPO_ROOT}/data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"
    output_folder = f"{REPO_ROOT}/data/{args.data_set}/RNA/FM_embeddings/BulkFormer/"

# Configuration
device = 'cuda'
graph_path = 'data/G_gtex.pt'
weights_path = 'data/G_gtex_weight.pt'
gene_emb_path = 'data/esm2_feature_concat.pt'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def symbols_to_ensg(symbols, species="human"):
    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        symbols,
        scopes="symbol,alias,prev_symbol",
        fields="ensembl.gene,symbol,name",
        species=("9606" if species=="human" else "10090"),
        as_dataframe=True,
        returnall=False
    )
    df = hits.reset_index().rename(columns={"query":"symbol"})

    # mygene may return dict/list with version suffixes (e.g., ENSG000001234.5)
    def pick_ensg(x):
        if isinstance(x, list):
            # list of dicts or strings
            for v in x:
                gid = (v.get("gene") if isinstance(v, dict) else v)
                if gid and gid.startswith(("ENSG","ENSMUSG")):
                    return gid.split(".")[0]
            return pd.NA
        if isinstance(x, dict):
            gid = x.get("gene")
            return gid.split(".")[0] if gid else pd.NA
        if isinstance(x, str):
            return x.split(".")[0]
        return pd.NA

    df["ensg_id"] = df["ensembl.gene"].map(pick_ensg)
    out = df[["symbol","ensg_id","name"]].drop_duplicates(subset=["symbol"])
    return out

def pickle_data(data, path):

    with open(path, "wb") as f: 
        pickle.dump(data, f)
def normalize_data(X_df, gene_length_dict):
    """
    Normalize RNA-seq count data to log-transformed TPM values.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix where rows represent samples and columns represent genes.
        Each entry contains the raw read count of a gene in a given sample.

    gene_length_dict : dict
        A dictionary mapping gene identifiers (Ensembl gene IDs) to gene lengths (in base pairs).

    Returns
    -------
    log_tpm_df : pandas.DataFrame
        A DataFrame of the same shape as `X_df`, containing log-transformed TPM values
        (i.e., log(TPM + 1)) for each gene in each sample.

    Description
    -----------
    This function converts raw RNA-seq count data to transcripts per million (TPM) values by
    normalizing for gene length and sample-specific total expression. Gene lengths are provided
    via `gene_length_dict`, and genes not present in the dictionary are assigned a default
    length of 1,000 bp (equivalent to no correction). The resulting TPM values are subsequently
    log-transformed using the natural logarithm (log1p). This normalization procedure accounts
    for both gene length and sequencing depth, facilitating cross-sample and cross-gene comparisons.
    """
    gene_names = X_df.columns
    gene_lengths_kb = np.array([gene_length_dict.get(gene, 1000) / 1000  for gene in gene_names])
    counts_matirx = X_df.values
    rate = counts_matirx / gene_lengths_kb
    sum_per_sample = rate.sum(axis=1)
    sum_per_sample[sum_per_sample == 0] = 1e-6  
    sum_per_sample = sum_per_sample.reshape(-1, 1)
    tpm = rate / sum_per_sample * 1e6
    log_tpm = np.log1p(tpm)
    log_tpm_df = pd.DataFrame(log_tpm,index=X_df.index, columns=X_df.columns)
    return log_tpm_df
def main_gene_selection(X_df, gene_list):
    """
    Aligns a gene expression matrix to a predefined gene list by adding placeholder values
    for missing genes and generating a binary mask indicating imputed entries.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix with rows representing samples and columns representing genes.
        The entries are typically log-transformed or normalized expression values.

    gene_list : list of str
        A predefined list of gene identifiers (Ensembl Gene IDs) to be retained
        in the final matrix. This list defines the desired gene space for downstream analyses.

    Returns
    -------
    X_df : pandas.DataFrame
        A gene expression matrix aligned to `gene_list`, with missing genes filled with a constant
        placeholder value (−10) and columns ordered accordingly.

    to_fill_columns : list of str
        A list of genes from `gene_list` that were not present in the original `X_df`
        and were therefore added with placeholder values.

    var : pandas.DataFrame
        A DataFrame with one row per gene, containing a binary column `'mask'` indicating
        whether a gene was imputed (1) or originally present (0). This can be used for masking
        in training or evaluation of models that distinguish observed and imputed entries.

    Notes
    -----
    This function ensures that all samples share a consistent gene space, which is essential
    for tasks such as model training, cross-dataset integration, or visualization. Placeholder
    values (−10) are used to maintain matrix shape while avoiding unintended bias in downstream
    statistical analyses or machine learning models.
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
 
    padding_df = pd.DataFrame(np.full((X_df.shape[0], len(to_fill_columns)), -10), 
                            columns=to_fill_columns, 
                            index=X_df.index)

    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var
def extract_feature(expr_array, 
                    high_var_gene_idx,
                    feature_type,
                    aggregate_type,
                    device,
                    batch_size,
                    return_expr_value = False,
                    esm2_emb = None,
                    valid_gene_idx = None):
    """
    Extracts transcriptome-level or gene-level feature representations from input expression profiles
    using a pre-trained deep learning model.

    Parameters
    ----------
    expr_array : np.ndarray
        A NumPy array of shape [N_samples, N_genes] representing gene expression profiles
        (e.g., log-transformed TPM values).

    high_var_gene_idx : list or np.ndarray
        Indices of highly variable genes used for transcriptome-level embedding aggregation.

    feature_type : str
        Specifies the type of feature to extract. Options:
            - 'transcriptome_level': aggregate gene embeddings to a single sample-level vector.
            - 'gene_level': retain per-gene embeddings for downstream fusion with external embeddings (e.g., ESM2).

    aggregate_type : str
        Aggregation method used when `feature_type='transcriptome_level'`. Options include:
            - 'max': use maximum value across selected genes.
            - 'mean': use average value.
            - 'median': use median value.
            - 'all': combine all three strategies by summation.

    device : torch.device
        Computation device (e.g., 'cuda' or 'cpu') for model inference.

    batch_size : int
        Number of samples per batch during feature extraction.

    return_expr_value : bool, optional
        If True, return predicted gene expression values instead of extracted embeddings. Default is False.

    esm2_emb : torch.Tensor, optional
        Precomputed ESM2 embeddings for all genes, used in gene-level feature concatenation.
        Required if `feature_type='gene_level'`.

    valid_gene_idx : list or np.ndarray, optional
        Indices of valid genes to be retained in gene-level embedding extraction.

    Returns
    -------
    result_emb : torch.Tensor
        The extracted feature representations:
            - [N_samples, D] for transcriptome-level features.
            - [N_samples, N_genes, D_concat] for gene-level features with ESM2 concatenation.

    or (if `return_expr_value=True`)
    expr_predictions : np.ndarray
        Model-predicted expression profiles for all samples.

    Notes
    -----
    This function supports two types of transcriptomic representations:
    (1) transcriptome-level features derived by aggregating gene-level embeddings from a deep model, and
    (2) gene-level embeddings optionally fused with external protein-based features such as ESM2.
    This allows flexible integration of expression and sequence-based representations for downstream tasks
    such as drug response prediction, disease classification, or feature alignment in multi-modal settings.
    """

    expr_tensor = torch.tensor(expr_array,dtype=torch.float32,device=device)
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False) 
    model.eval()

    all_emb_list = []
    all_expr_value_list = []

    with torch.no_grad():
        if feature_type == 'transcriptome_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)
                output, emb = model(X, [2])
                all_expr_value_list.append(output.detach().cpu().numpy())
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:,high_var_gene_idx,:]
 
                if aggregate_type == 'max':
                    final_emb =np.max(emb_valid, axis=1)
                elif aggregate_type == 'mean':
                    final_emb =np.mean(emb_valid, axis=1)
                elif aggregate_type == 'median':
                    final_emb =np.median(emb_valid, axis=1)
                elif aggregate_type == 'all':
                    max_emb =np.max(emb_valid, axis=1)
                    mean_emb =np.mean(emb_valid, axis=1)
                    median_emb =np.median(emb_valid, axis=1)
                    final_emb = max_emb+mean_emb+median_emb

                all_emb_list.append(final_emb)
            result_emb = np.vstack(all_emb_list)
            result_emb = torch.tensor(result_emb,device='cpu',dtype=torch.float32)

        elif feature_type == 'gene_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)
                output, emb = model(X, [2])
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:,valid_gene_idx,:]
                all_emb_list.append(emb_valid)
                all_expr_value_list.append(output.detach().cpu().numpy())
            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb,device='cpu',dtype=torch.float32)
            esm2_emb_selected = esm2_emb[valid_gene_idx]
            esm2_emb_expanded = esm2_emb_selected.unsqueeze(0).expand(all_emb_tensor.shape[0], -1, -1) 
            esm2_emb_expanded = esm2_emb_expanded.to('cpu')

            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)
    
    if return_expr_value:
        return np.vstack(all_expr_value_list)
    
    else:
        return result_emb
# # Load demo normalized data (log-transformed TPM)
# log_tpm_df = pd.read_csv('data/demo_normalized_data.csv')

# # Load demo count data (raw count)
# count_df = pd.read_csv('data/demo_count_data.csv')

# # Convert raw counts to normalized expression values (log-transformed TPM)
# gene_length_df = pd.read_csv('data/gene_length_df.csv')
# gene_length_dict = gene_length_df.set_index('ensg_id')['length'].to_dict()
# log_tpm_df = normalize_data(X_df=count_df, gene_length_dict=gene_length_dict)
bulkformer_gene_info = pd.read_csv('data/bulkformer_gene_info.csv')
bulkformer_gene_list = bulkformer_gene_info['ensg_id'].to_list()

data = anndata.read_h5ad(data_path)
data = restrict_to_samples(data, args.data_set, args.sample_ids)

# Pick IDs to embed (aliquots for TARGET_AML, patient-level otherwise)
sample_ids = sample_ids_of(data, args.data_set)

# Build expression df with gene symbols as columns
data_df = pd.DataFrame(
    data.X,
    index=sample_ids,
    columns=data.var["gene_symbol"].values
)
print("Data shape before sieving: ", data_df.shape)

# Gene symbol -> Ensembl mapping. mygene answers change over time, so the mapping used for a
# dataset is saved on first use and reused afterwards (delete the file to re-query).
mapping_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bulkformer_mygene_maps", f"{args.data_set}.csv")
if os.path.exists(mapping_cache):
    pairs = pd.read_csv(mapping_cache, keep_default_na=False)  # keep symbols such as "NA" as text
    print(f"Using cached symbol->Ensembl mapping: {mapping_cache}")
else:
    mapping = symbols_to_ensg(data_df.columns)
    mapping["ensg_id"] = mapping["ensg_id"].replace({"<NA>": pd.NA})
    mapping = mapping.dropna(subset=["ensg_id"])
    mapping = mapping[mapping["ensg_id"].isin(bulkformer_gene_list)]
    pairs = pd.DataFrame({"symbol": [c[0] for c in mapping["symbol"].values.tolist()],
                          "ensg_id": mapping["ensg_id"].values})
    os.makedirs(os.path.dirname(mapping_cache), exist_ok=True)
    pairs.to_csv(mapping_cache, index=False)
    print(f"Saved symbol->Ensembl mapping to {mapping_cache}")

data_df_sieved = data_df.loc[:, pairs["symbol"].astype(str).tolist()]
data_df_sieved.columns = pairs["ensg_id"].tolist()
data_df_sieved = data_df_sieved.groupby(data_df_sieved.columns, axis=1).mean()

print("Data shape after sieving: ", data_df_sieved.shape)

# Align expression data to a predefined gene list with placeholder imputation for missing genes.
input_df, to_fill_columns, var = main_gene_selection(X_df=data_df_sieved, gene_list=bulkformer_gene_list)

print(f"Genes present in BulkFormer list: {len(bulkformer_gene_list) - len(to_fill_columns)} / {len(bulkformer_gene_list)}")
print(f"Genes filled with -10 (missing): {len(to_fill_columns)} / {len(bulkformer_gene_list)}")
print(f"Coverage: {(len(bulkformer_gene_list) - len(to_fill_columns)) / len(bulkformer_gene_list):.1%}")

# Save stats about gene coverage and input data quality for reference
stats = {
    "dataset": args.data_set, "fm": "BulkFormer",
    "input_shape": list(input_df.shape),
    "genes_present": len(bulkformer_gene_list) - len(to_fill_columns),
    "genes_missing": len(to_fill_columns),
    "total_genes": len(bulkformer_gene_list),
    "coverage": (len(bulkformer_gene_list) - len(to_fill_columns)) / len(bulkformer_gene_list),
    "non_neg10_fraction": float((input_df.values > -10).mean()),
}

if args.just_preprocess_data:
    with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved preprocessing stats to JSON.")
    exit(0)

var.reset_index(inplace=True)
valid_gene_idx = list(var[var["mask"] == 0].index)

high_var_gene_idx = torch.load("data/high_var_gene_list.pt", weights_only=False)

# Initialize the BulkFormer model with preloaded graph structure and gene embeddings.
graph = torch.load(graph_path, map_location='cpu', weights_only=False)
weights = torch.load(weights_path, map_location='cpu', weights_only=False)
graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)
gene_emb = torch.load(gene_emb_path, map_location='cpu', weights_only=False)
model_params['graph'] = graph
model_params['gene_emb'] = gene_emb
model = BulkFormer(**model_params).to(device)
# Load the pretrained BulkFormer model checkpoint for inference or fine-tuning.
ckpt_model = torch.load('model/Bulkformer_ckpt_epoch_29.pt',weights_only=False)

new_state_dict = OrderedDict()
for key, value in ckpt_model.items():
    new_key = key[7:] if key.startswith("module.") else key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

for i, sid in tqdm(enumerate(sample_ids), total=len(sample_ids)):
    # Always overwrite: skipping existing files silently kept stale embeddings on reruns
    output_path = os.path.join(output_folder, f"{sid}.pkl")

    emb = extract_feature(
        expr_array=input_df.values[i:i+1],
        high_var_gene_idx=high_var_gene_idx,
        feature_type="transcriptome_level",
        aggregate_type="max",
        device=device,
        batch_size=4,
        return_expr_value=False,
        esm2_emb=model_params["gene_emb"],
        valid_gene_idx=valid_gene_idx,
    ).cpu().numpy()[0]

    assert emb.shape[0] != 0
    pickle_data(emb, output_path)

'''
# Build expression df with gene symbols as columns

data_df = pd.DataFrame(
    data.X,
    index=data.obs[obs_key].values,
    columns=data.var["gene_symbol"].values
)

print("Data shape before sieving: ", data_df.shape)

mapping = symbols_to_ensg(data_df.columns)

mapping["ensg_id"] = mapping["ensg_id"].replace({"<NA>": pd.NA})
mapping = mapping.dropna(subset=["ensg_id"])
mapping = mapping[mapping["ensg_id"].isin(bulkformer_gene_list)]
data_df_sieved = data_df.loc[:, [c[0] for c in mapping["symbol"].values.tolist()]]
data_df_sieved.columns = mapping["ensg_id"]
data_df_sieved = data_df_sieved.groupby(data_df_sieved.columns, axis=1).mean()

print("Data shape after sieving: ", data_df_sieved.shape)

patient_id = data.obs[obs_key].values.tolist()

# Align expression data to a predefined gene list with placeholder imputation for missing genes.
input_df , to_fill_columns, var= main_gene_selection(X_df=data_df_sieved,gene_list=bulkformer_gene_list)
var.reset_index(inplace=True)
valid_gene_idx = list(var[var['mask'] == 0].index)
high_var_gene_idx = torch.load('data/high_var_gene_list.pt',weights_only=False)
for i, id_ in tqdm(enumerate(patient_id)):

    output_path = os.path.join(output_folder, f"{id_}.pkl")
    if not os.path.exists(output_path):

        # Extract transcritome-level embedding
        emb = extract_feature(
            expr_array= input_df.values[i:i+1],
            high_var_gene_idx=high_var_gene_idx,
            feature_type='transcriptome_level',
            aggregate_type='max',
            device=device,
            batch_size=4,
            return_expr_value=False,
            esm2_emb=model_params['gene_emb'],
            valid_gene_idx=valid_gene_idx
        ).cpu().numpy()[0]

        assert emb.shape[0] != 0 
        pickle_data(emb, output_path)

'''
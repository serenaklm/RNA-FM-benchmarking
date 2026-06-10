import os
import anndata
import pickle
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import scgpt as scg
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import argparse
import json
import sys

def pickle_data(data, path): 

    with open(path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    
        
    warnings.filterwarnings("ignore")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate scGPT embeddings')
    parser.add_argument('--data_set', type=str, required=True,
                    help='Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...)')
    parser.add_argument('--just_preprocess_data', action='store_true')
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Override data directory. If set, reads {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad '
             'and writes to {data_dir}/RNA/FM_embeddings/scGPT/.',
    )
    args = parser.parse_args()

    # Set paths based on dataset
    # All log1p transformed
    if args.data_dir:
        output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "scGPT")
        counts_path   = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
    else:
        output_folder = f"../../data/{args.data_set}/RNA/FM_embeddings/scGPT"
        counts_path   = f"../../data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"

    model_folder = "../../data/scGPT/scGPT_human"
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    gene_col = "gene_symbol"
    cell_type_key = "cell_type"

    # Tutorial: https://scgpt.readthedocs.io/en/latest/tutorial_reference_mapping.html
    
    RNAseq_info = sc.read_h5ad(counts_path)
    print(RNAseq_info.to_df().shape)
    RNAseq_info.var_names = [gene.split('.')[0] for gene in RNAseq_info.var_names]

    N_HVG = 5000 # scGPT typically selects HVGs as a pre-processing step

    sc.pp.highly_variable_genes(RNAseq_info, n_top_genes=N_HVG, flavor='seurat_v3')
    RNAseq_hvg = RNAseq_info[:, RNAseq_info.var['highly_variable']]
    
    expr_array = RNAseq_hvg.X
    if hasattr(expr_array, 'toarray'):
        expr_array = expr_array.toarray()

    print("Gene expression stats after HVG selection:")
    print(f"  Shape: {expr_array.shape}")
    print(f"  Non-zero fraction: {(expr_array > 0).mean():.3f}")
    print(f"  Mean: {expr_array.mean():.4f}, Max: {expr_array.max():.4f}")
    print(f"  Zero-filled columns (missing genes): {(expr_array == 0).all(axis=0).sum()} / {expr_array.shape[1]}")
    print(f"  Matched genes: {RNAseq_hvg.n_vars} / {N_HVG}")

    stats = {
        "dataset": args.data_set, "fm": "scGPT",
        "input_shape": list(expr_array.shape),
        "non_zero_fraction": float((expr_array > 0).mean()),
        "mean": float(expr_array.mean()),
        "max": float(expr_array.max()),
        "zero_filled_genes": int((expr_array == 0).all(axis=0).sum()),
        "total_genes": int(N_HVG),
        "matched_genes": int(RNAseq_hvg.n_vars),
    }

    if args.just_preprocess_data:
        with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
            json.dump(stats, f, indent=4)
        print("Saved preprocessing stats to JSON.")
        exit(0)

    # Get embeddings
    RNAseq_hvg_scGPT_emb = scg.tasks.embed_data(
        RNAseq_hvg,
        model_folder,
        gene_col=gene_col,
        batch_size=64,
        return_new_adata=True,
    )
    
'''
    obs = 'cases.submitter_id' if args.data_set == 'TCGA' else 'patient_id'
    patient_ids = RNAseq_info.obs[obs].values.tolist()
    RNAseq_hvg_scGPT_emb = RNAseq_hvg_scGPT_emb.to_df().values
    
    for i, id_ in tqdm(enumerate(patient_ids)):

        output_path = os.path.join(output_folder, f"{id_}.pkl")
        emb = RNAseq_hvg_scGPT_emb[i]
        pickle_data(emb, output_path)
'''

# Choose IDs to save by
if args.data_set == "TARGET_AML":
    ids = RNAseq_info.obs_names.astype(str).tolist()
elif args.data_set == "TCGA":
    ids = RNAseq_info.obs['cases.submitter_id'].astype(str).tolist()
elif 'patient_id' in RNAseq_info.obs.columns:
    ids = RNAseq_info.obs['patient_id'].astype(str).tolist()
else:
    # ICI cohorts: obs_names are the patient/sample IDs
    ids = RNAseq_info.obs_names.astype(str).tolist()

print(f"Computing embeddings for {len(ids)} patients/samples")
print("Sanmple IDs:", ids[:5])

RNAseq_hvg_scGPT_emb = RNAseq_hvg_scGPT_emb.to_df().values

for i, id_ in tqdm(enumerate(ids), total=len(ids)):
    output_path = os.path.join(output_folder, f"{id_}.pkl")
    emb = RNAseq_hvg_scGPT_emb[i]
    pickle_data(emb, output_path)
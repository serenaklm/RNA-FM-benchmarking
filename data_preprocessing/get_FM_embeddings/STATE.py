import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path
import subprocess, shlex
from tqdm import tqdm
import argparse
import json

from fm_common import add_subset_args, restrict_to_samples, sample_ids_of

os.environ.setdefault("HF_HOME", "/data/rbg/users/ashyamal/huggingface_cache")

import torch
from huggingface_hub import snapshot_download

def pickle_data(data, path): 

    with open(path, "wb") as f:
        pickle.dump(data, f)
        
# Get args
parser = argparse.ArgumentParser(description='Generate STATE embeddings')
parser.add_argument('--data_set', type=str, required=True,
                help='Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...)')
parser.add_argument('--action', type=str, required=True, choices=['process_hvg', 'extract_embeddings'],
                    help='Action to perform: process_hvg or extract_embeddings')
parser.add_argument('--just_preprocess_data', action='store_true')
parser.add_argument(
    '--data_dir', type=str, default=None,
    help='Override data directory. If set, reads/writes from {data_dir}/RNA/.',
)
add_subset_args(parser, hvg=True)
args = parser.parse_args()

print(f"Using data set: {args.data_set}")
print("Action:", args.action)

if args.action == 'process_hvg':

    # Get the genes that are captured in state's protein embeddings
    protein_emb_path = "../../data/STATE/SE-600M/protein_embeddings.pt"
    protein_emb = torch.load(protein_emb_path)
    captured_genes = list(protein_emb.keys())

    key = f"{args.data_set}_bulk_RNA_processed_log1p"
    if args.data_dir:
        data_folder = os.path.join(args.data_dir, "RNA")
    else:
        data_folder = f"../../data/{args.data_set}/RNA"

    RNAseq_info = ad.read_h5ad(os.path.join(data_folder, f"{key}.h5ad"))
    data_output_path = os.path.join(data_folder, f"{key}_HVG.h5ad")
    
    print(f"Input has {RNAseq_info.n_obs} samples x {RNAseq_info.n_vars} genes")
    print("Sanmple IDs:", RNAseq_info.obs_names.astype(str)[:5].tolist())

    N_HVG = 5000
    # HVGs are selected on the whole cohort, or on --hvg_reference to keep a frozen list
    hvg_source = ad.read_h5ad(args.hvg_reference) if args.hvg_reference else RNAseq_info
    assert list(hvg_source.var_names) == list(RNAseq_info.var_names), "reference genes differ"
    sc.pp.highly_variable_genes(hvg_source, n_top_genes=N_HVG, flavor='seurat_v3')
    selected_genes = hvg_source.var['highly_variable'].index[hvg_source.var['highly_variable']].values.tolist()
    RNAseq_info = restrict_to_samples(RNAseq_info, args.data_set, args.sample_ids)
    selected_genes = [g for g in selected_genes if g in captured_genes]
    print(f"There are {len(selected_genes)}")
    RNAseq_hvg = RNAseq_info[:, selected_genes]
    
    if args.data_dir:
        output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "STATE") + "/"
    else:
        output_folder = f"../../data/{args.data_set}/RNA/FM_embeddings/STATE/"
    os.makedirs(output_folder, exist_ok=True)

    expr_array = RNAseq_hvg.X
    if hasattr(expr_array, 'toarray'):
        expr_array = expr_array.toarray()

    print("Gene expression stats after HVG selection:")
    print(f"  Shape: {expr_array.shape}")
    print(f"  Non-zero fraction: {(expr_array > 0).mean():.3f}")
    print(f"  Mean: {expr_array.mean():.4f}, Max: {expr_array.max():.4f}")
    print(f"  Zero-filled columns (missing genes): {(expr_array == 0).all(axis=0).sum()} / {expr_array.shape[1]}")
    print(f"  Matched genes: {len(selected_genes)} / {len(captured_genes)}")

    stats = {
        "dataset": args.data_set, "fm": "STATE",
        "input_shape": list(expr_array.shape),
        "non_zero_fraction": float((expr_array > 0).mean()),
        "mean": float(expr_array.mean()),
        "max": float(expr_array.max()),
        "zero_filled_genes": int((expr_array == 0).all(axis=0).sum()),
        "total_genes": int(len(captured_genes)),
        "matched_genes": int(len(selected_genes)),
    }

    if args.just_preprocess_data:
        with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
            json.dump(stats, f, indent=4)
        print("Saved preprocessing stats to JSON.")
        exit(0)
    
    RNAseq_hvg.write(data_output_path)
    
elif args.action == 'extract_embeddings':
    if args.data_dir:
        output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "STATE") + "/"
    else:
        output_folder = f"../../data/{args.data_set}/RNA/FM_embeddings/STATE/"

    STATE_emb_info = ad.read_h5ad(os.path.join(output_folder, f"{args.data_set}_STATE_emb.h5ad"))
    STATE_emb = STATE_emb_info.obsm["X_state"]

    # Choose IDs to save by
    ids = sample_ids_of(STATE_emb_info, args.data_set)

    print(f"Computing embeddings for {len(ids)} patients/samples")
    print("Sanmple IDs:", ids[:5])

    for i, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Always overwrite: a skip-if-exists here silently kept stale embeddings on reruns
        current_output_path = os.path.join(output_folder, f"{id_}.pkl")
        emb = STATE_emb[i]            # same as STATE_emb[i:i+1][0], simpler
        pickle_data(emb, current_output_path)

'''
Training command:

state emb transform \
  --model-folder <REPO_ROOT>/data/STATE/SE-600M \
  --input <REPO_ROOT>/data/CALGB/RNA/CALGB_40603_bulk_RNA_processed_log1p_HVG.h5ad \
  --output <REPO_ROOT>/data/CALGB/RNA/FM_embeddings/STATE/CALGB_STATE_emb.h5ad

'''
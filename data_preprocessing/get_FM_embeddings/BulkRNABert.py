import os
import pickle
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from fm_common import add_subset_args, restrict_to_samples, sample_ids_of
import json

os.environ.setdefault("HF_HOME", "/data/rbg/users/ashyamal/huggingface_cache")

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def pickle_data(data, path): 

    with open(path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, required=True,
                    help='Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...)')
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument('--just_preprocess_data', action='store_true')
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Override data directory. If set, reads {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad '
             'and writes to {data_dir}/RNA/FM_embeddings/BulkRNABert/.',
    )

    add_subset_args(parser)
    args = parser.parse_args()

    # All log1p transformed
    if args.data_dir:
        RNA_counts_path = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
        output_folder   = os.path.join(args.data_dir, "RNA", "FM_embeddings", "BulkRNABert")
    else:
        RNA_counts_path = f"../../data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"
        output_folder   = f"../../data/{args.data_set}/RNA/FM_embeddings/BulkRNABert"

    bulkbertgenes_path = "../../data/BulkRNABert/common_gene_id.txt"
    genecode_path = "../../data/GENCODE/gene_length.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Tutorial: https://huggingface.co/InstaDeepAI/BulkRNABert 

    # Load in the data 
    genes = [g.replace("\n", "").strip() for g in open(bulkbertgenes_path)]
    RNAseq_info = anndata.read_h5ad(RNA_counts_path)
    RNAseq_info = restrict_to_samples(RNAseq_info, args.data_set, args.sample_ids)
    gencode = pd.read_csv(genecode_path)
    gene_symbol, gene_id = gencode["gene_name"].values.tolist(), gencode["gene_id"].values.tolist()

    mapping = {gene_symbol[i]: gene_id[i].split(".")[0] for i in range(len(gene_symbol))}

    RNA_counts = RNAseq_info.to_df()
    RNA_counts.columns = [mapping.get(c, "-") for c in RNA_counts.columns]
    print(RNA_counts.shape)
    RNA_counts = RNA_counts.loc[:, [c != "-" for c in RNA_counts.columns]]
    print(RNA_counts.shape)
    RNA = RNA_counts.reindex(columns=list(genes), fill_value=0.0)
    
    '''
    obs = 'cases.submitter_id' if args.data_set == 'TCGA' else 'patient_id'
    patient_id = RNAseq_info.obs.loc[:, obs].values
    print(f"There are {len(patient_id)} patients", "df is of shape", RNA.shape)
    '''
    
    ids = sample_ids_of(RNAseq_info, args.data_set)
        
    print(f"Computing embeddings for {len(ids)} patients/samples")
    print("Sample IDs:", ids[:5])
    
    if not args.just_preprocess_data:
        # Device
        device = torch.device("cpu")

        # Load model and tokenizer
        config = AutoConfig.from_pretrained(
            "InstaDeepAI/BulkRNABert",
            trust_remote_code=True,
        )
        config.embeddings_layers_to_save = (4,) # last transformer layer

        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/BulkRNABert", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            "InstaDeepAI/BulkRNABert",
            config=config,
            trust_remote_code=True,
        ).to(device)

    # Load bulk RNA-seq data and preprocess them.
    gene_expression_array = RNA.to_numpy()
    gene_expression_array = gene_expression_array / np.log(10) # gene_expression_array originally = ln(TPM + 1), convert to log10
    
    print("Gene expression stats after normalization:")
    print(f"  Shape: {gene_expression_array.shape}")
    print(f"  Non-zero fraction: {(gene_expression_array > 0).mean():.3f}")
    print(f"  Mean: {gene_expression_array.mean():.4f}, Max: {gene_expression_array.max():.4f}")
    print(f"  Zero-filled columns (missing genes): {(gene_expression_array == 0).all(axis=0).sum()} / {gene_expression_array.shape[1]}")

    # Check how many genes were matched vs zero-filled
    matched = RNA_counts.columns.isin(genes).sum()
    print(f"  Genes matched to BulkRNABert vocab: {matched} / {len(genes)}")
    
    stats = {
        "dataset": args.data_set, "fm": "BulkRNABert",
        "input_shape": list(gene_expression_array.shape),
        "non_zero_fraction": float((gene_expression_array > 0).mean()),
        "mean": float(gene_expression_array.mean()),
        "max": float(gene_expression_array.max()),
        "zero_filled_genes": int((gene_expression_array == 0).all(axis=0).sum()),
        "total_genes": int(gene_expression_array.shape[1]),
        "matched_genes": int(matched),
    }
    
    if args.just_preprocess_data:
        with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
            json.dump(stats, f, indent=4)
        print("Saved preprocessing stats to JSON.")
        exit(0)

    assert gene_expression_array.shape[1] == config.n_genes
    gene_expression_ids = tokenizer.batch_encode_plus(gene_expression_array, return_tensors="pt")["input_ids"]

    """
    # Inference
    for i, id_ in tqdm(enumerate(patient_id)):

        current_output_path = os.path.join(output_folder, f"{id_}.pkl")
        
        if not os.path.exists(current_output_path):
            emb = model(gene_expression_ids[i:i+1, :].to(device))["embeddings_4"].mean(axis=1)
            pickle_data(emb, current_output_path)
    """ 
    
    #for i, id_ in tqdm(enumerate(patient_id)):
    for i, id_ in tqdm(enumerate(ids), total=len(ids)):
        if (i % args.num_shards) != args.shard_id:
            continue

        current_output_path = os.path.join(output_folder, f"{id_}.pkl")

        # Skip existing files to resume sharded runs, unless re-embedding listed samples
        if args.sample_ids or not os.path.exists(current_output_path):
            # Inference only: without no_grad, activations over the ~19k-gene sequence are kept
            # for backprop (60-97 GB per process). Embedding values are unchanged.
            with torch.no_grad():
                emb = model(gene_expression_ids[i:i+1, :].to(device))["embeddings_4"].mean(axis=1)
            pickle_data(emb, current_output_path)
import os
import pickle
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
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
                        choices=['TCGA', 'CALGB', 'CPTAC', 'Hartwig', 'ICGC', 'TARGET_AML'])
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument('--just_preprocess_data', action='store_true')

    args = parser.parse_args()

    # Paths
    RNA_counts_path = f"../../data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"
    bulkbertgenes_path = "../../data/BulkRNABert/common_gene_id.txt"
    genecode_path = "../../data/GENCODE/gene_length.csv"

    # NEW OUTPUT ROOT
    output_root = f"../../data/{args.data_set}/RNA/FM_embeddings/BulkRNABert_HVG"

    HVG_SIZES = [1250, 5000, 10000]

    for k in HVG_SIZES:
        os.makedirs(os.path.join(output_root, f"{k}_HVGs"), exist_ok=True)

    # Load data
    genes = [g.strip() for g in open(bulkbertgenes_path)]
    RNAseq_info = anndata.read_h5ad(RNA_counts_path)
    gencode = pd.read_csv(genecode_path)

    gene_symbol = gencode["gene_name"].values.tolist()
    gene_id = gencode["gene_id"].values.tolist()
    mapping = {gene_symbol[i]: gene_id[i].split(".")[0] for i in range(len(gene_symbol))}

    RNA_counts = RNAseq_info.to_df()
    RNA_counts.columns = [mapping.get(c, "-") for c in RNA_counts.columns]
    RNA_counts = RNA_counts.loc[:, [c != "-" for c in RNA_counts.columns]]

    RNA = RNA_counts.reindex(columns=list(genes), fill_value=0.0)

    # IDs
    if args.data_set == "TARGET_AML":
        ids = RNAseq_info.obs_names.astype(str).tolist()
    else:
        obs = 'cases.submitter_id' if args.data_set == 'TCGA' else 'patient_id'
        ids = RNAseq_info.obs.loc[:, obs].astype(str).tolist()

    print(f"Computing embeddings for {len(ids)} samples")

    # =============================
    # 🔥 Compute HVG indices ONCE
    # =============================

    orig_gene_symbols = RNAseq_info.var_names.astype(str).tolist()
    mapped_gene_ids = [mapping.get(g, "-") for g in orig_gene_symbols]
    lv = RNAseq_info.var["log_variance"].to_numpy()

    hvg_df = pd.DataFrame({
        "mapped_gene_id": mapped_gene_ids,
        "log_variance": lv,
    })

    hvg_df = hvg_df[hvg_df["mapped_gene_id"] != "-"].copy()
    hvg_df = (
        hvg_df.sort_values("log_variance", ascending=False)
              .drop_duplicates(subset="mapped_gene_id", keep="first")
    )

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    hvg_df = hvg_df[hvg_df["mapped_gene_id"].isin(gene_to_idx)]

    hvg_indices_dict = {}
    for k in HVG_SIZES:
        top_ids = hvg_df.nlargest(k, "log_variance")["mapped_gene_id"].tolist()
        hvg_indices_dict[k] = [gene_to_idx[g] for g in top_ids]
        print(f"{k} HVGs mapped:", len(hvg_indices_dict[k]))

    # =============================
    # Model
    # =============================
    if not args.just_preprocess_data:
        device = torch.device("cpu")

        config = AutoConfig.from_pretrained(
            "InstaDeepAI/BulkRNABert",
            trust_remote_code=True,
        )
        config.embeddings_layers_to_save = (4,)

        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/BulkRNABert",
            trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
            "InstaDeepAI/BulkRNABert",
            config=config,
            trust_remote_code=True,
        ).to(device)

    # Preprocess expression
    gene_expression_array = RNA.to_numpy()
    gene_expression_array = gene_expression_array / np.log(10)

    assert gene_expression_array.shape[1] == config.n_genes

    gene_expression_ids = tokenizer.batch_encode_plus(
        gene_expression_array,
        return_tensors="pt"
    )["input_ids"]

    # =============================
    # Inference loop
    # =============================

    for i, id_ in tqdm(enumerate(ids), total=len(ids)):

        if (i % args.num_shards) != args.shard_id:
            continue

        # Check if ALL outputs exist → skip
        skip = True
        for k in HVG_SIZES:
            path = os.path.join(output_root, f"{k}_HVGs", f"{id_}.pkl")
            if not os.path.exists(path):
                skip = False
                break
        if skip:
            continue

        with torch.no_grad():
            out = model(gene_expression_ids[i:i+1, :].to(device))["embeddings_4"]
            gene_emb = out.squeeze(0).cpu().numpy()  # [G, D]

        # Compute and save each HVG pooling
        for k in HVG_SIZES:
            out_path = os.path.join(output_root, f"{k}_HVGs", f"{id_}.pkl")

            if os.path.exists(out_path):
                continue

            hvg_emb = gene_emb[hvg_indices_dict[k]].mean(axis=0).astype(np.float32)

            pickle_data(hvg_emb, out_path)
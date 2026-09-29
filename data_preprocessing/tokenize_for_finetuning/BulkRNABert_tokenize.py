"""
Pre-tokenize RNA-seq data for BulkRNABert fine-tuning.

Saves a single pickle {patient_id: np.int8 array of shape (n_genes,)} so the
tokenizer only has to run once, rather than at every DataLoader __getitem__.

Run from: data_preprocessing/tokenize_for_finetuning/
Example:
  python BulkRNABert_tokenize.py --data_set TCGA
  python BulkRNABert_tokenize.py --data_set CPTAC
"""

import os
import pickle
import argparse

import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("HF_HOME", "/data/rbg/users/ashyamal/huggingface_cache")

from transformers import AutoConfig, AutoTokenizer


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        choices=["TCGA", "CPTAC", "CALGB", "Hartwig", "ICGC", "TARGET_AML"],
    )
    args = parser.parse_args()
    dataset = args.data_set

    # ---- paths (relative to this script's location) ----
    RNA_counts_path = f"../../data/{dataset}/RNA/{dataset}_bulk_RNA_processed_log1p.h5ad"
    bulkbertgenes_path = "../../data/BulkRNABert/common_gene_id.txt"
    genecode_path = "../../data/GENCODE/gene_length.csv"
    output_dir = "../../data/tokens/BulkRNABert"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_tokens.pkl")

    # ---- load gene vocab ----
    genes = [g.replace("\n", "").strip() for g in open(bulkbertgenes_path)]
    print(f"BulkRNABert vocab size: {len(genes)} genes")

    # ---- load RNA data ----
    RNAseq_info = anndata.read_h5ad(RNA_counts_path)
    gencode = pd.read_csv(genecode_path)
    gene_symbol = gencode["gene_name"].values.tolist()
    gene_id = gencode["gene_id"].values.tolist()
    mapping = {gene_symbol[i]: gene_id[i].split(".")[0] for i in range(len(gene_symbol))}

    RNA_counts = RNAseq_info.to_df()
    RNA_counts.columns = [mapping.get(c, "-") for c in RNA_counts.columns]
    RNA_counts = RNA_counts.loc[:, [c != "-" for c in RNA_counts.columns]]
    RNA = RNA_counts.reindex(columns=list(genes), fill_value=0.0)

    # ---- patient IDs ----
    if dataset == "TARGET_AML":
        ids = RNAseq_info.obs_names.astype(str).tolist()
    else:
        obs = "cases.submitter_id" if dataset == "TCGA" else "patient_id"
        ids = RNAseq_info.obs.loc[:, obs].astype(str).tolist()

    print(f"Tokenizing {len(ids)} patients for {dataset}")

    # ---- preprocess: ln(TPM+1) → log10(TPM+1) ----
    gene_expression_array = RNA.to_numpy() / np.log(10)
    print(f"Expression stats: min={gene_expression_array.min():.3f}, max={gene_expression_array.max():.3f}")

    # ---- load tokenizer ----
    config = AutoConfig.from_pretrained("InstaDeepAI/BulkRNABert", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/BulkRNABert", trust_remote_code=True)

    assert gene_expression_array.shape[1] == config.n_genes, (
        f"Gene count mismatch: got {gene_expression_array.shape[1]}, expected {config.n_genes}"
    )

    # ---- tokenize all at once ----
    print("Running tokenizer...")
    token_ids = tokenizer.batch_encode_plus(gene_expression_array, return_tensors=None)["input_ids"]
    # token_ids is a list of lists of ints, each in [0, 63]
    token_ids = np.array(token_ids, dtype=np.int8)  # (n_patients, n_genes)

    # ---- save as dict {patient_id: int8 array} ----
    tokens_dict = {pid: token_ids[i] for i, pid in tqdm(enumerate(ids), total=len(ids))}
    save_pickle(tokens_dict, output_path)

    print(f"Saved {len(tokens_dict)} tokenized samples to {output_path}")
    print(f"Array dtype: {token_ids.dtype}, shape per patient: {token_ids.shape[1:]}")

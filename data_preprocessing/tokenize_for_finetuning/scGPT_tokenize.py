"""
Pre-tokenize RNA-seq data for scGPT fine-tuning.

Replicates the preprocessing done inside scgpt.tasks.embed_data / get_batch_cell_embeddings:
  1. Map gene symbols to scGPT vocab IDs, keep only in-vocab genes.
  2. For each patient: select nonzero-expression genes, prepend <cls> token.
  3. Store raw log1p expression values (binning is deferred to the DataLoader
     collate_fn to match exactly what embed_data does at inference time).

Saves a single pickle:
    {patient_id: {"gene_ids": np.int64 array, "expressions": np.float32 array}}

Both arrays are variable-length (no padding here).

Run from: data_preprocessing/tokenize_for_finetuning/
Example:
  python scGPT_tokenize.py --data_set TCGA
  python scGPT_tokenize.py --data_set CPTAC
"""

import os
import pickle
import argparse

import anndata
import numpy as np
from tqdm import tqdm

# scgpt_env only
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from scgpt.tokenizer import GeneVocab


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

    # ---- paths ----
    RNA_counts_path = f"../../data/{dataset}/RNA/{dataset}_bulk_RNA_processed_log1p.h5ad"
    model_dir = "../../data/scGPT/scGPT_human"
    vocab_file = os.path.join(model_dir, "vocab.json")
    output_dir = "../../data/tokens/scGPT"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset}_tokens.pkl")

    # ---- load vocab ----
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    cls_id = vocab["<cls>"]
    pad_value = -2  # matches args.json pad_value

    print(f"scGPT vocab size: {len(vocab)}")

    # ---- load RNA data ----
    adata = anndata.read_h5ad(RNA_counts_path)
    print(f"Loaded adata: {adata.shape}")

    # strip Ensembl version suffix if present (e.g. ENSG00000001234.5 → ENSG00000001234)
    adata.var_names = [g.split(".")[0] for g in adata.var_names]

    # map each gene to its scGPT vocab ID (-1 if absent)
    gene_symbols = adata.var_names.tolist()
    id_in_vocab = np.array(
        [vocab[g] if g in vocab else -1 for g in gene_symbols], dtype=np.int64
    )
    in_vocab_mask = id_in_vocab >= 0
    n_matched = int(in_vocab_mask.sum())
    print(f"Genes in scGPT vocab: {n_matched} / {len(gene_symbols)}")

    # filter adata to in-vocab genes
    adata = adata[:, in_vocab_mask]
    gene_ids_in_vocab = id_in_vocab[in_vocab_mask]  # (n_in_vocab,) int64

    count_matrix = adata.X
    if hasattr(count_matrix, "toarray"):
        count_matrix = count_matrix.toarray()
    count_matrix = count_matrix.astype(np.float32)  # (n_patients, n_in_vocab)

    # ---- patient IDs ----
    if dataset == "TARGET_AML":
        ids = adata.obs_names.astype(str).tolist()
    else:
        obs = "cases.submitter_id" if dataset == "TCGA" else "patient_id"
        ids = adata.obs[obs].astype(str).tolist()

    print(f"Tokenizing {len(ids)} patients for {dataset}")

    # ---- tokenize each patient ----
    tokens_dict = {}
    for i, pid in tqdm(enumerate(ids), total=len(ids)):
        row = count_matrix[i]  # (n_in_vocab,)

        # keep nonzero genes (same as get_batch_cell_embeddings)
        nonzero_idx = np.nonzero(row)[0]
        gene_ids_sample = gene_ids_in_vocab[nonzero_idx]    # vocab IDs
        expr_sample = row[nonzero_idx]                       # raw log1p values

        # prepend <cls> token (expression gets pad_value at cls position)
        gene_ids_sample = np.concatenate([[cls_id], gene_ids_sample]).astype(np.int64)
        expr_sample = np.concatenate([[pad_value], expr_sample]).astype(np.float32)

        tokens_dict[pid] = {
            "gene_ids": gene_ids_sample,
            "expressions": expr_sample,
        }

    save_pickle(tokens_dict, output_path)

    # report stats
    lengths = [len(v["gene_ids"]) for v in tokens_dict.values()]
    print(f"Saved {len(tokens_dict)} tokenized samples to {output_path}")
    print(f"Sequence length (incl. cls): min={min(lengths)}, "
          f"median={int(np.median(lengths))}, max={max(lengths)}")

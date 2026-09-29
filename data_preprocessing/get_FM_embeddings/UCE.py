"""
Generate UCE (Universal Cell Embeddings) embeddings for bulk RNA-seq samples.

Each patient/sample is treated as a single "cell". This mirrors the pipeline in
the UCE repo's eval_single_anndata.py / evaluate.py
(UCE_REPO_DIR), reusing UCE's own model, gene-embedding
loading, and cell-sentence sampling code directly. The one thing NOT reused is
UCE's disk-memmap-backed multi-dataset Dataset/DataLoader machinery
(MultiDatasetSentences) -- that exists to stream millions of single cells off
disk, which is unnecessary for our hundreds-of-samples bulk cohorts, so
cell-sentence sampling is done directly on an in-memory tensor instead.

UCE expects raw (un-normalized) counts: it builds a per-cell, expression-
weighted gene sampling distribution internally via log1p(counts) (see
sample_cell_sentences below). Our stored input, like the other FM scripts in
this folder, is already log1p-transformed, so we undo that (expm1) before
handing counts to UCE rather than double-log-transforming.
"""
import os
import sys
import shutil
import pickle
import warnings
import argparse
import json
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import torch
from torch import nn
from tqdm import tqdm

from fm_common import add_subset_args, restrict_to_samples, sample_ids_of

REPO_ROOT = Path(__file__).resolve().parents[2]
UCE_REPO_DIR = os.environ.get("UCE_REPO_DIR", "/data/rbg/users/ashyamal/UCE")
UCE_DATA_DIR = os.environ.get("UCE_DATA_DIR", str(REPO_ROOT / "data" / "UCE"))
UCE_MODEL_FILES_DIR = os.path.join(UCE_DATA_DIR, "model_files")

# Captured before the os.chdir below, so relative CLI args (e.g. --data_dir)
# can still be resolved against the directory the script was actually invoked
# from, rather than silently resolving against UCE_DATA_DIR instead.
_INVOCATION_CWD = os.getcwd()

sys.path.insert(0, UCE_REPO_DIR)

# data_proc/gene_embeddings.py reads "model_files/new_species_protein_embeddings.csv"
# relative to the cwd at IMPORT time, and data_proc/data_utils.py's
# get_species_to_pe() reads the same relative path at CALL time -- so the cwd
# needs to be UCE_DATA_DIR (where our copy of model_files/ lives) before either
# of those happen.
os.makedirs(UCE_MODEL_FILES_DIR, exist_ok=True)
new_species_csv = os.path.join(UCE_MODEL_FILES_DIR, "new_species_protein_embeddings.csv")
if not os.path.exists(new_species_csv):
    shutil.copy(os.path.join(UCE_REPO_DIR, "model_files", "new_species_protein_embeddings.csv"), new_species_csv)
os.chdir(UCE_DATA_DIR)

from utils import figshare_download                                        # UCE/utils.py
from model import TransformerModel                                         # UCE/model.py
from data_proc.gene_embeddings import load_gene_embeddings_adata           # UCE/data_proc/gene_embeddings.py
from data_proc.data_utils import (                                         # UCE/data_proc/data_utils.py
    get_species_to_pe,
    get_spec_chrom_csv,
    adata_path_to_prot_chrom_starts,
)


def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def ensure_support_files(args):
    """Download UCE's shared support files if not already present.
    Copied from UCE/evaluate.py AnndataProcessor.check_paths(), minus the
    model-weight and sample-adata downloads (we always supply our own)."""
    figshare_download("https://figshare.com/ndownloader/files/42706558", args.spec_chrom_csv_path)
    figshare_download("https://figshare.com/ndownloader/files/42706555", args.offset_pkl_path)
    if not os.path.exists(args.protein_embeddings_dir):
        figshare_download("https://figshare.com/ndownloader/files/42715213",
                           os.path.join(UCE_MODEL_FILES_DIR, "protein_embeddings.tar.gz"))
    figshare_download("https://figshare.com/ndownloader/files/42706585", args.token_file)


# ---- Copied/adapted from UCE/eval_data.py: sample_cell_sentences ----
# Same token-ordering logic as upstream (CLS token, then genes grouped by
# chromosome and ordered by genomic start, chrom order shuffled per cell),
# just operating on an in-memory counts/weights tensor for a single dataset
# instead of reading from UCE's disk-memmap multi-dataset structures.
def sample_cell_sentences(counts, batch_weights, gene_pe_idxs, chroms, starts, args):
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))
    mask = torch.zeros((counts.shape[0], args.pad_length))

    for c in range(counts.shape[0]):
        weights = batch_weights[c].numpy()
        weights = weights / weights.sum()  # RE NORM after mask

        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = np.random.choice(np.arange(len(weights)), size=args.sample_size, p=weights, replace=True)
        choosen_chrom = chroms[choice_idx]  # get the sampled genes' chromosomes
        chrom_sort = np.argsort(choosen_chrom)  # order the genes by chromosome
        choice_idx = choice_idx[chrom_sort]

        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((args.pad_length), args.cls_token_idx)  # start with cls
        i = 1  # continue on to the rest of the sequence with left bracket being assumed
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms)  # chroms have no natural order, shuffle them

        for chrom in uq_chroms:
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET  # open chrom token
            i += 1
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(choosen_starts[loc])  # sort genes by start within this chrom

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = gene_pe_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx  # close chrom token
            i += 1

        remainder_len = args.pad_length - i
        cell_mask = torch.concat((torch.ones(i), torch.zeros(remainder_len)))
        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = args.pad_token_idx
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)

    return cell_sentences.long(), mask


# ---- Copied from UCE/evaluate.py ----
def get_ESM2_embeddings(args):
    all_pe = torch.load(args.token_file)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS))
        all_pe.requires_grad = False
    return all_pe


def build_model(args):
    # Copied from UCE/evaluate.py run_eval model setup
    emsize = 1280  # embedding dimension
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05
    model = TransformerModel(
        token_dim=args.token_dim, d_model=emsize, nhead=nhead,
        d_hid=args.d_hid, nlayers=args.nlayers, dropout=dropout,
        output_dim=args.output_dim,
    )
    # intialize as empty
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(args.model_loc, map_location="cpu"), strict=True)

    # Load in the real token embeddings
    all_pe = get_ESM2_embeddings(args)
    if all_pe.shape[0] != 145469:
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    print(f"Loaded model:\n{args.model_loc}")
    return model.eval()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate UCE embeddings')
    parser.add_argument('--data_set', type=str, required=True,
                         help='Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...)')
    parser.add_argument('--just_preprocess_data', action='store_true')
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Override data directory. If set, reads {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad '
             'and writes to {data_dir}/RNA/FM_embeddings/UCE/.',
    )
    parser.add_argument('--species', type=str, default='human')
    add_subset_args(parser)

    # Model arguments -- default to the 33-layer model from the UCE paper
    parser.add_argument('--model_loc', type=str, default=os.path.join(UCE_MODEL_FILES_DIR, "33layer_model.torch"))
    parser.add_argument('--nlayers', type=int, default=33, help='Number of transformer layers.')
    parser.add_argument('--output_dim', type=int, default=1280, help='Output embedding dimension.')
    parser.add_argument('--d_hid', type=int, default=5120, help='Hidden dimension.')
    parser.add_argument('--token_dim', type=int, default=5120, help='Token dimension.')
    parser.add_argument('--batch_size', type=int, default=25,
                         help='Per-GPU batch size. UCE recommends 25 for the 33-layer model on an 80GB GPU.')
    parser.add_argument('--pad_length', type=int, default=1536, help='Length to pad the cell sentence to.')
    parser.add_argument('--sample_size', type=int, default=1024, help='Number of genes sampled per cell sentence.')
    parser.add_argument('--pad_token_idx', type=int, default=0)
    parser.add_argument('--chrom_token_left_idx', type=int, default=1)
    parser.add_argument('--chrom_token_right_idx', type=int, default=2)
    parser.add_argument('--cls_token_idx', type=int, default=3)
    parser.add_argument('--CHROM_TOKEN_OFFSET', type=int, default=143574)

    # Shared UCE support-file locations (downloaded once, shared across datasets)
    parser.add_argument('--spec_chrom_csv_path', type=str, default=os.path.join(UCE_MODEL_FILES_DIR, "species_chrom.csv"))
    parser.add_argument('--token_file', type=str, default=os.path.join(UCE_MODEL_FILES_DIR, "all_tokens.torch"))
    parser.add_argument('--protein_embeddings_dir', type=str, default=os.path.join(UCE_MODEL_FILES_DIR, "protein_embeddings"))
    parser.add_argument('--offset_pkl_path', type=str, default=os.path.join(UCE_MODEL_FILES_DIR, "species_offsets.pkl"))

    args = parser.parse_args()
    # UCE samples genes per cell (np.random.choice, chromosome order shuffle); seed so that
    # reruns give the same embeddings.
    np.random.seed(0)
    torch.manual_seed(0)

    # A relative --data_dir is relative to the directory the script was
    # invoked from (_INVOCATION_CWD), not the UCE_DATA_DIR we already chdir'd
    # into above.
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(_INVOCATION_CWD, args.data_dir)

    # Set paths based on dataset -- same convention as the other FM scripts
    if args.data_dir:
        output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "UCE")
        counts_path = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
    else:
        output_folder = f"{REPO_ROOT}/data/{args.data_set}/RNA/FM_embeddings/UCE"
        counts_path = f"{REPO_ROOT}/data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"

    os.makedirs(output_folder, exist_ok=True)

    RNAseq_info = sc.read_h5ad(counts_path)
    RNAseq_info = restrict_to_samples(RNAseq_info, args.data_set, args.sample_ids)
    print(RNAseq_info.to_df().shape)
    RNAseq_info.var_names = [gene.split('.')[0] for gene in RNAseq_info.var_names]

    # Subset to genes with ESM2 protein embeddings -- this is UCE's gene
    # vocabulary, playing the same role that N_HVG selection plays for scGPT
    # or the captured_genes intersection plays for STATE.
    ensure_support_files(args)
    RNAseq_uce, protein_embeddings = load_gene_embeddings_adata(
        adata=RNAseq_info, species=[args.species], embedding_model="ESM2",
    )

    expr_array = RNAseq_uce.X
    if hasattr(expr_array, 'toarray'):
        expr_array = expr_array.toarray()

    print("Gene expression stats after subsetting to UCE's gene vocabulary:")
    print(f"  Shape: {expr_array.shape}")
    print(f"  Non-zero fraction: {(expr_array > 0).mean():.3f}")
    print(f"  Mean: {expr_array.mean():.4f}, Max: {expr_array.max():.4f}")
    print(f"  Matched genes: {RNAseq_uce.n_vars} / {RNAseq_info.n_vars}")

    stats = {
        "dataset": args.data_set, "fm": "UCE",
        "input_shape": list(expr_array.shape),
        "non_zero_fraction": float((expr_array > 0).mean()),
        "mean": float(expr_array.mean()),
        "max": float(expr_array.max()),
        "total_genes": int(RNAseq_info.n_vars),
        "matched_genes": int(RNAseq_uce.n_vars),
    }

    if args.just_preprocess_data:
        with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
            json.dump(stats, f, indent=4)
        print("Saved preprocessing stats to JSON.")
        sys.exit(0)

    # Per-gene lookup needed to build UCE's cell sentences: row index into the
    # ESM2 token embedding table, chromosome, and genomic start position.
    with open(args.offset_pkl_path, "rb") as f:
        species_to_offsets = pickle.load(f)
    species_to_pe = get_species_to_pe(args.protein_embeddings_dir)
    spec_pe_genes = list(species_to_pe[args.species].keys())
    offset = species_to_offsets[args.species]
    gene_to_chrom_pos = get_spec_chrom_csv(args.spec_chrom_csv_path)

    gene_pe_idxs, chroms, starts = adata_path_to_prot_chrom_starts(
        RNAseq_uce, args.species, spec_pe_genes, gene_to_chrom_pos, offset,
    )
    gene_pe_idxs = gene_pe_idxs.numpy()
    chroms = chroms.values
    starts = np.asarray(starts)

    # Undo the log1p applied when this h5ad was created, then re-derive UCE's
    # expression-weighted sampling distribution the same way UCE does from raw
    # counts (see run_eval / MultiDatasetSentences in UCE/evaluate.py, UCE/eval_data.py)
    counts = torch.from_numpy(np.expm1(expr_array)).float().clamp(min=0)
    weights = torch.log1p(counts)

    model = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_embeds = []
    n = counts.shape[0]
    with torch.no_grad():
        for start_i in tqdm(range(0, n, args.batch_size)):
            end_i = min(start_i + args.batch_size, n)
            batch_counts = counts[start_i:end_i]
            batch_weights = weights[start_i:end_i]

            batch_sentences, mask = sample_cell_sentences(
                batch_counts, batch_weights, gene_pe_idxs, chroms, starts, args,
            )
            batch_sentences = batch_sentences.to(device)
            mask = mask.to(device)

            batch_sentences = batch_sentences.permute(1, 0)
            batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)  # Normalize token outputs now
            _, embedding = model.forward(batch_sentences, mask=mask)
            all_embeds.append(embedding.detach().cpu().numpy())

    RNAseq_uce_emb = np.vstack(all_embeds)

    # Choose IDs to save by
    ids = sample_ids_of(RNAseq_info, args.data_set)

    print(f"Computing embeddings for {len(ids)} patients/samples")
    print("Sanmple IDs:", ids[:5])

    for i, id_ in tqdm(enumerate(ids), total=len(ids)):
        output_path = os.path.join(output_folder, f"{id_}.pkl")
        emb = RNAseq_uce_emb[i]
        pickle_data(emb, output_path)

    print(f"Saved embeddings to {output_folder}")

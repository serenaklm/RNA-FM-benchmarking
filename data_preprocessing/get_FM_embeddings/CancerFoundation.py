"""
CancerFoundation embeddings. Copied into this repo from the CancerFoundation checkout
(CANCERFOUNDATION_DIR/tutorial/CancerFoundation.py) so the benchmark's
embedding code is versioned here. Runs with the checkout's tutorial/ folder as working
directory, since the model code and weights are loaded relative to it.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CF_TUTORIAL_DIR = os.path.join(os.environ.get("CANCERFOUNDATION_DIR", "/data/rbg/users/ashyamal/CancerFoundation"), "tutorial")
_INVOCATION_CWD = os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(CF_TUTORIAL_DIR)
sys.path.insert(0, "../")
import scanpy as sc
from model.embedding import embed
import os
import pickle
from tqdm import tqdm
import argparse
import json
import numpy as np

import model.embedding as cf_embedding
from fm_common import add_subset_args, restrict_to_samples, sample_ids_of
import scipy.sparse

# Model and data paths
model_dir = "../model/assets"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate CancerFoundation embeddings')
parser.add_argument('--data_set', type=str, required=True,
                help='Dataset to process (e.g. TCGA, CALGB, CPTAC, Hartwig, ICGC, TARGET_AML, Hugo, Riaz, Rose, ...)')
parser.add_argument('--just_preprocess_data', action='store_true')
parser.add_argument(
    '--data_dir', type=str, default=None,
    help='Override data directory. If set, reads {data_dir}/RNA/{data_set}_bulk_RNA_processed_log1p.h5ad '
         'and writes to {data_dir}/RNA/FM_embeddings/CancerFoundation/.',
)
add_subset_args(parser, hvg=True)
args = parser.parse_args()
for _a in ("data_dir", "sample_ids", "hvg_reference"):
    if getattr(args, _a) and not os.path.isabs(getattr(args, _a)):
        setattr(args, _a, os.path.join(_INVOCATION_CWD, getattr(args, _a)))

# Set paths based on dataset
if args.data_dir:
    output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "CancerFoundation")
    counts_path = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed_log1p.h5ad")
else:
    output_folder = f"{REPO_ROOT}/data/{args.data_set}/RNA/FM_embeddings/CancerFoundation"
    counts_path = f"{REPO_ROOT}/data/{args.data_set}/RNA/{args.data_set}_bulk_RNA_processed_log1p.h5ad"

# Load data
print(f"Loading {args.data_set} data...")
adata = sc.read_h5ad(counts_path)
print(f"Loaded {adata.n_obs} samples x {adata.n_vars} genes")

# Set keys based on your AnnData structure
#batch_key = "project.project_id"  # Cancer type for batch correction (TCGA-BRCA, etc.)
#bio_key = "project.project_id"     # For visualization

if args.data_set == 'TCGA':
    batch_key = "project.project_id"  # Cancer type for batch correction (TCGA-BRCA, etc.)
    
else:
    adata.obs['batch_dummy'] = f'{args.data_set}_Study'
    batch_key = 'batch_dummy'
    
# obs key determined after loading data (see ID extraction below)

expr_array = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)

print("Gene expression stats:")
print(f"  Shape: {expr_array.shape}")
print(f"  Non-zero fraction: {(expr_array > 0).mean():.3f}")
print(f"  Mean: {expr_array.mean():.4f}, Max: {expr_array.max():.4f}")
print(f"  Zero-filled columns (missing genes): {(expr_array == 0).all(axis=0).sum()} / {expr_array.shape[1]}")
print(f"  Matched genes: {adata.n_vars} / {adata.n_vars} (CF aligns internally)")

stats = {
    "dataset": args.data_set, "fm": "CancerFoundation",
    "input_shape": list(expr_array.shape),
    "non_zero_fraction": float((expr_array > 0).mean()),
    "mean": float(expr_array.mean()),
    "max": float(expr_array.max()),
    "zero_filled_genes": int((expr_array == 0).all(axis=0).sum()),
    "total_genes": int(adata.n_vars),
    "matched_genes": int(adata.n_vars),
}

os.makedirs(output_folder, exist_ok=True)
if args.just_preprocess_data:
    with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved preprocessing stats to JSON.")
    exit(0)

if args.hvg_reference:
    # Frozen HVG list: reproduce embed()'s own selection (vocabulary filter, then
    # cell_ranger HVGs with the same batch key) on the reference data, and make embed()
    # use that list instead of re-selecting HVGs on the input.
    from pathlib import Path
    ref = sc.read_h5ad(args.hvg_reference)
    assert list(ref.var_names) == list(adata.var_names), "reference genes differ"
    if batch_key == "batch_dummy":
        ref.obs["batch_dummy"] = f"{args.data_set}_Study"
    vocab = cf_embedding.GeneVocab.from_file(Path(model_dir) / "vocab.json")
    ref = ref[:, [g in vocab for g in ref.var_names]].copy()
    sc.pp.highly_variable_genes(ref, n_top_genes=1200 - 1, flavor="cell_ranger", batch_key=batch_key)
    frozen_hvgs = set(ref.var_names[ref.var["highly_variable"]])
    print(f"Frozen HVG list from {args.hvg_reference}: {len(frozen_hvgs)} genes")

    def _frozen_hvg_selection(a, **kwargs):
        a.var["highly_variable"] = a.var_names.isin(frozen_hvgs)

    cf_embedding.sc.pp.highly_variable_genes = _frozen_hvg_selection

adata = restrict_to_samples(adata, args.data_set, args.sample_ids)

# Get embeddings
print("Generating CancerFoundation embeddings...")
embed_adata = embed(
    adata_or_file=adata,
    model_dir=model_dir,
    batch_key=batch_key,
    batch_size=64,
)

# Choose IDs to save by
ids = sample_ids_of(embed_adata, args.data_set)

print(f"Computing embeddings for {len(ids)} patients/samples")
print("Sanmple IDs:", ids[:5])

# Save embeddings as individual pickle files

embeddings = embed_adata.obsm["CancerGPT"]  # Extract embeddings

for idx, id_ in tqdm(enumerate(ids), total=len(ids)):
    output_path = os.path.join(output_folder, f"{id_}.pkl")
    embedding = embeddings[idx]  # Already numpy array
    
    with open(output_path, 'wb') as f:
        pickle.dump(embedding, f)

print(f"Saved {len(ids)} embeddings to {output_folder}")
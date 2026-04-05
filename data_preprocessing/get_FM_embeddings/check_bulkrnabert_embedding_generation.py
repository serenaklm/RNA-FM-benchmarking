"""
Round-trip test for BulkRNABert gene symbol -> ENSG mapping.

tcga_sample.csv (from HuggingFace BulkRNABert) has ENSG IDs as column headers
and expression values already in log10(TPM+1) scale ready for the model.

Our pipeline (BulkRNABert.py) starts from gene symbols in h5ad, maps them to
ENSG IDs via GENCODE, aligns to BulkRNABert's vocab, then divides by log(10)
to convert from ln(TPM+1) to log10(TPM+1).

This test checks:
1. How many ENSG IDs in tcga_sample.csv round-trip correctly through our mapping
2. Whether the final aligned expression arrays are identical
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TCGA_SAMPLE_PATH = os.path.join(SCRIPT_DIR, "tcga_sample.csv")
VOCAB_PATH       = os.path.join(SCRIPT_DIR, "../../data/BulkRNABert/common_gene_id.txt")
GENCODE_PATH     = os.path.join(SCRIPT_DIR, "../../data/GENCODE/gene_length.csv")

# ---------------------------------------------------------------------------
# Load reference data
# ---------------------------------------------------------------------------
print("Loading BulkRNABert vocab...")
genes = [g.strip() for g in open(VOCAB_PATH)]
print(f"  Vocab size: {len(genes)}")

print("Loading GENCODE mapping...")
gencode = pd.read_csv(GENCODE_PATH)
symbol_list = gencode["gene_name"].values.tolist()
id_list     = gencode["gene_id"].values.tolist()

# Forward and reverse GENCODE maps (strip version suffixes, same as BulkRNABert.py)
symbol_to_ensg = {symbol_list[i]: id_list[i].split(".")[0] for i in range(len(symbol_list))}
ensg_to_symbol = {id_list[i].split(".")[0]: symbol_list[i] for i in range(len(id_list))}

print("Loading tcga_sample.csv...")
tcga = pd.read_csv(TCGA_SAMPLE_PATH, index_col=0)
# Strip version suffixes from ENSG IDs in case they have them (e.g. ENSG00000001.5)
tcga.columns = [c.split(".")[0] for c in tcga.columns]
# Ensure all values are numeric
tcga = tcga.apply(pd.to_numeric, errors='coerce').fillna(0.0)
print(f"  Shape: {tcga.shape}  (samples x genes)")
print(f"  First 5 column names: {tcga.columns[:5].tolist()}")
print(f"  Value range: [{tcga.values.min():.3f}, {tcga.values.max():.3f}]  (expected: log10 scale ~0-4)")

# ---------------------------------------------------------------------------
# REFERENCE PATH
# tcga_sample.csv has raw TPM values. Official code does np.log10(1 + x).
# Align to vocab first (fill missing with 0), then apply log10(1+x).
# ---------------------------------------------------------------------------
RNA_direct_raw = tcga.reindex(columns=genes, fill_value=0.0)
RNA_direct = np.log10(1 + RNA_direct_raw.to_numpy())
n_direct_matched = tcga.columns.isin(genes).sum()
print(f"\n[Reference] ENSG IDs in tcga_sample.csv present in vocab: {n_direct_matched} / {len(genes)}")

# ---------------------------------------------------------------------------
# PIPELINE PATH
# Simulate what BulkRNABert.py does:
#   h5ad has gene symbols + ln(TPM+1) values
#   -> map symbol to ENSG via GENCODE
#   -> reindex to vocab (fill_value=0)
#   -> divide by log(10) to get log10(TPM+1)
#
# To simulate from tcga_sample.csv (raw TPM):
#   - apply log1p to get ln(TPM+1) — this is what h5ad stores
#   - rename ENSG columns to gene symbols (reverse GENCODE)
#   - apply forward mapping (symbol -> ENSG)
#   - deduplicate ENSG columns by taking mean (handles symbols that map to same ENSG)
#   - reindex to vocab
#   - divide by log(10)
# ---------------------------------------------------------------------------

# Step 1: simulate ln-scale h5ad values
tcga_ln = np.log1p(tcga)

# Step 2: reverse-map ENSG IDs -> gene symbols
reverse_cols = [ensg_to_symbol.get(e, "-") for e in tcga_ln.columns]
n_no_symbol = sum(1 for c in reverse_cols if c == "-")
print(f"\n[Pipeline] ENSG IDs in tcga_sample.csv with no symbol in GENCODE: {n_no_symbol} / {len(tcga.columns)}")

tcga_as_symbols = tcga_ln.copy()
tcga_as_symbols.columns = reverse_cols
tcga_as_symbols = tcga_as_symbols.loc[:, tcga_as_symbols.columns != "-"]

# Step 3: forward map gene symbols -> ENSG (exactly as BulkRNABert.py)
tcga_as_symbols.columns = [symbol_to_ensg.get(c, "-") for c in tcga_as_symbols.columns]
tcga_as_symbols = tcga_as_symbols.loc[:, tcga_as_symbols.columns != "-"]

# Step 4: deduplicate ENSG columns (multiple symbols can map to same ENSG)
tcga_as_symbols = tcga_as_symbols.groupby(tcga_as_symbols.columns, axis=1).mean()

# Step 5: align to vocab and convert to log10(TPM+1)
RNA_pipeline = tcga_as_symbols.reindex(columns=genes, fill_value=0.0) / np.log(10)

# ---------------------------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------------------------
arr_d = RNA_direct                      # already numpy (log10 scale)
arr_p = RNA_pipeline.to_numpy()         # log10 scale

# Gene-level: which ENSG positions are populated (non-zero) in each path
direct_populated   = set(genes[i] for i in range(len(genes)) if arr_d[:, i].any())
pipeline_populated = set(genes[i] for i in range(len(genes)) if arr_p[:, i].any())

only_direct   = direct_populated - pipeline_populated
only_pipeline = pipeline_populated - direct_populated
shared        = direct_populated & pipeline_populated

print("\n=== Gene coverage comparison ===")
print(f"  Genes populated in both paths:     {len(shared)}")
print(f"  Only in reference (direct):        {len(only_direct)}")
print(f"  Only in pipeline:                  {len(only_pipeline)}")

if only_direct:
    print(f"\n  ENSG IDs present in tcga_sample.csv but lost in pipeline (first 10):")
    for e in list(only_direct)[:10]:
        sym = ensg_to_symbol.get(e, "NOT IN GENCODE")
        remapped = symbol_to_ensg.get(sym, "N/A")
        print(f"    {e}  ->  symbol: {sym}  ->  re-mapped ENSG: {remapped}")

if only_pipeline:
    print(f"\n  ENSG IDs appearing only in pipeline (first 10) — possible swaps:")
    for e in list(only_pipeline)[:10]:
        sym = ensg_to_symbol.get(e, "NOT IN GENCODE")
        print(f"    {e}  (symbol: {sym})")

print("\n=== Value comparison (shared genes only) ===")
shared_idx = [i for i, g in enumerate(genes) if g in shared]
arr_d_shared = arr_d[:, shared_idx]
arr_p_shared = arr_p[:, shared_idx]

max_diff  = float(np.abs(arr_d_shared - arr_p_shared).max())
mean_diff = float(np.abs(arr_d_shared - arr_p_shared).mean())
match     = np.allclose(arr_d_shared, arr_p_shared, atol=1e-5)

print(f"  Arrays match (atol=1e-5): {match}")
print(f"  Max absolute difference:  {max_diff:.8f}")
print(f"  Mean absolute difference: {mean_diff:.8f}")

if not match:
    mismatch_col_idx = np.where(~np.isclose(arr_d_shared, arr_p_shared, atol=1e-5).all(axis=0))[0]
    print(f"  Mismatched gene positions: {len(mismatch_col_idx)}")
    print(f"  First 5 mismatched ENSG IDs: {[genes[shared_idx[i]] for i in mismatch_col_idx[:5]]}")
else:
    print("  Gene symbol -> ENSG mapping is correct for all shared genes.")

print("\n=== Summary ===")
print(f"  tcga_sample.csv genes in vocab:  {n_direct_mapped} / {len(genes)}" if False else
      f"  tcga_sample.csv genes in vocab:  {n_direct_matched} / {len(genes)}")
print(f"  Genes surviving round-trip:      {len(shared)} / {n_direct_matched}")
print(f"  Genes lost in round-trip:        {len(only_direct)}")
print(f"  Value arrays match:              {match}")

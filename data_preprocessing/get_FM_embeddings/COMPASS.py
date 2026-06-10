print("Importing libraries...")
import compass
from compass import PreTrainer, FineTuner, loadcompass
import anndata as ad
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import argparse
import json

# Utils
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Set data arguments: CALGB vs TCGA using argparse --data_set
parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, required=True,
                help='Dataset to process (e.g. TCGA, CPTAC, Hugo, Riaz, Rose, ...)')
parser.add_argument('--just_preprocess_data', action='store_true')
parser.add_argument(
    '--data_dir', type=str, default=None,
    help='Override data directory (e.g. for ICI cohorts). '
         'If set, reads {data_dir}/RNA/{data_set}_bulk_RNA_processed.h5ad '
         'and writes embeddings to {data_dir}/RNA/FM_embeddings/COMPASS/.',
)
parser.add_argument(
    '--cancer_code', type=int, default=None,
    help='Integer COMPASS cancer code (overrides per-dataset lookup). '
         'Required for ICI cohorts. See data/COMPASS/cancer_code.json.',
)
args = parser.parse_args()

# No log-transformed
adata_path = {
    "TCGA": "../../data/TCGA/RNA/TCGA_bulk_RNA_raw.h5ad",
    "CALGB": "../../data/CALGB/RNA/CALGB_bulk_RNA_processed.h5ad",
    "CPTAC": "../../data/CPTAC/RNA/CPTAC_bulk_RNA_processed.h5ad",
    "Hartwig": "../../data/Hartwig/RNA/Hartwig_bulk_RNA_processed.h5ad",
    "ICGC": "../../data/ICGC/RNA/ICGC_bulk_RNA_processed.h5ad",
    "TARGET_AML": "../../data/TARGET_AML/RNA/TARGET_AML_bulk_RNA_processed.h5ad"
}

if args.data_dir:
    raw_h5ad = os.path.join(args.data_dir, "RNA", f"{args.data_set}_bulk_RNA_processed.h5ad")
elif args.data_set in adata_path:
    raw_h5ad = adata_path[args.data_set]
else:
    raise ValueError(f"Unknown data_set '{args.data_set}' and no --data_dir provided.")

print(f"Reading {args.data_set} data from {raw_h5ad}...")
adata = ad.read_h5ad(raw_h5ad)

data_sets_with_cancer_info = {
    "CPTAC": "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/CPTAC/obs_to_tcga_mapping.pkl",
    "ICGC": "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/ICGC/id_to_tcga_mapping.pkl",
    "Hartwig": "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/Hartwig/id_to_tcga_mapping.pkl"
}

if args.data_set in data_sets_with_cancer_info:
    id_to_tcga = load_pickle(data_sets_with_cancer_info[args.data_set])
    obs_to_use = id_to_tcga.keys()
    print("Original number of patients:", adata.n_obs)
    adata = adata[adata.obs['patient_id'].isin(obs_to_use)].copy()
    print("Filtered number of patients with dataset-to-TCGA mapping:", adata.n_obs)

# Get cancer mapping from https://www.immuno-compass.com/download/other/cancer_code.json
cancer_mapping = {"ACC":0,"BLCA":1,"BRCA":2,"CESC":3,"CHOL":4,"COAD":5,"DLBC":6,"ESCA":7,"GBM":8,"HNSC":9,"KICH":10,"KIRC":11,"KIRP":12,"LAML":13,"LGG":14,"LIHC":15,"LUAD":16,"LUSC":17,"MESO":18,"OV":19,"PAAD":20,"PCPG":21,"PRAD":22,"READ":23,"SARC":24,"SKCM":25,"STAD":26,"TGCT":27,"THCA":28,"THYM":29,"UCEC":30,"UCS":31,"UVM":32}

'''
df = pd.DataFrame(
    adata.X,  # numpy array with expression values
    columns=adata.var['gene_symbol'].values  # gene symbols as column names
)
'''

# Choose IDs to embed/save by
if args.data_set == "TARGET_AML":
    ids = adata.obs_names.astype(str).tolist()
elif args.data_set == "TCGA":
    ids = adata.obs['cases.submitter_id'].astype(str).tolist()
elif 'patient_id' in adata.obs.columns:
    ids = adata.obs['patient_id'].astype(str).tolist()
else:
    # ICI cohorts: obs_names are the patient/sample IDs
    ids = adata.obs_names.astype(str).tolist()

df = pd.DataFrame(
    adata.X,
    index=ids,  # <-- important
    columns=adata.var['gene_symbol'].values
)

print("Read dataframe with shape:", df.shape)

# Add patient barcode as first column
obs = 'cases.submitter_id' if args.data_set == 'TCGA' else 'patient_id'
#df.insert(0, 'bcr_patient_barcode', adata.obs[obs].values)
df.insert(0, 'bcr_patient_barcode', ids)

# Add cancer code as second column
if args.cancer_code is not None:
    # Explicit override (e.g. for ICI cohorts passed via --cancer_code)
    df.insert(1, 'cancer_code', args.cancer_code)

elif args.data_set == 'TCGA':
    cancer_types = adata.obs['project.project_id'].str.replace('TCGA-', '', regex=False)
    df.insert(1, 'cancer_code', cancer_types.map(cancer_mapping).values)

elif args.data_set == 'CALGB':
    df.insert(1, 'cancer_code', 2)

elif args.data_set in data_sets_with_cancer_info:
    cancer_types = adata.obs['patient_id'].map(id_to_tcga)
    df.insert(1, 'cancer_code', cancer_types.map(cancer_mapping).values)

elif args.data_set == 'TARGET_AML':
    df.insert(1, 'cancer_code', 13)

else:
    raise ValueError(
        f"Unknown data set '{args.data_set}' and no --cancer_code provided. "
        "For ICI cohorts pass --cancer_code <int>."
    )

gene_map = pd.read_csv("../../data/COMPASS/compass_gene_map.csv")
compass_genes = gene_map['gene_name'].tolist()

#df_aligned = df[['bcr_patient_barcode', 'cancer_code'] + compass_genes].copy()
# Alignment
target_columns = ['bcr_patient_barcode', 'cancer_code'] + compass_genes
df_aligned = df.reindex(columns=target_columns, fill_value=0.0)
missing_genes = set(compass_genes) - set(df.columns)
print(f"Added {len(missing_genes)} missing genes as zero-columns.")
# Set index to patient barcode
df_for_compass = df_aligned.set_index('bcr_patient_barcode')

print("Formatted input RNA dataframe with shape:", df_for_compass.shape)

# Create output directory
if args.data_dir:
    output_folder = os.path.join(args.data_dir, "RNA", "FM_embeddings", "COMPASS")
else:
    output_folder = f"../../data/{args.data_set}/RNA/FM_embeddings/COMPASS/"
os.makedirs(output_folder, exist_ok=True)

expr_array = df_for_compass[compass_genes].values
matched_genes_count = len(compass_genes) - len(missing_genes)

print("Gene expression stats:")
print(f"  Shape: {expr_array.shape}")
print(f"  Non-zero fraction: {(expr_array > 0).mean():.3f}")
print(f"  Mean: {expr_array.mean():.4f}, Max: {expr_array.max():.4f}")
print(f"  Zero-filled columns (missing genes): {(expr_array == 0).all(axis=0).sum()} / {expr_array.shape[1]}")
print(f"  Matched genes: {matched_genes_count} / {len(compass_genes)}")

stats = {
    "dataset": args.data_set, "fm": "COMPASS",
    "input_shape": list(expr_array.shape),
    "non_zero_fraction": float((expr_array > 0).mean()),
    "mean": float(expr_array.mean()),
    "max": float(expr_array.max()),
    "zero_filled_genes": int((expr_array == 0).all(axis=0).sum()),
    "total_genes": int(len(compass_genes)),
    "matched_genes": int(matched_genes_count),
}

if args.just_preprocess_data:
    with open(os.path.join(output_folder, f"preprocessing_stats_{args.data_set}.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved preprocessing stats to JSON.")
    exit(0)
    
# Load model and extract embeddings
print("Loading model...")
model = loadcompass('../../data/COMPASS/pretrainer.pt')

print("Extracting embeddings...")
dfgn, dfgs, dfct = model.extract(df_for_compass, batch_size=128, with_gene_level=True)
print("Done")

embeddings_df = dfct
'''
if 'bcr_patient_barcode' in embeddings_df.columns:
    patient_ids = embeddings_df['bcr_patient_barcode'].values
else:
    patient_ids = embeddings_df.index.values
'''
# Get embedding columns (exclude patient ID column if present)
embedding_cols = [col for col in embeddings_df.columns if col != 'bcr_patient_barcode']

patient_ids = ids

print(f"Computing embeddings for {len(patient_ids)} patients/samples")
print("Sanmple IDs:", list(patient_ids)[:5])

# Save each patient's embedding
for idx, patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids)):
    output_path = os.path.join(output_folder, f"{patient_id}.pkl")
    
    # Get patient's embedding (as numpy array or tensor)
    # always index by row position
    embedding = embeddings_df.iloc[idx][embedding_cols].to_numpy()

    # enforce 1D embedding
    assert embedding.ndim == 1, f"{patient_id}: embedding has shape {embedding.shape}, expected (D,)"

    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(embedding, f)

print(f"Saved embeddings to {output_folder}")
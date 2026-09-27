"""
get_ICI_embeddings.py

Build processed pickle files for ICI cohorts, aligned to the TCGA gene vocabulary.

Usage (from repo root):
    # Build one cohort
    python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort Hugo
    python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort Riaz
    python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort Rose

    # Merge all cohorts listed in configs/immunotherapy_cohorts.yaml into one pkl
    python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge

Output:
    data/RNA_embeddings/ICI/{cohort}_processed.pkl   (per-cohort)
    data/RNA_embeddings/ICI/ICI_all_processed.pkl    (merged; required for LOCO/transfer splits)

Each sample dict contains:
    RNA_seq          : float32 array aligned to TCGA gene vocab (17490 genes)
    max_*/ssGSEA_*/PCA_*  : baseline embedding features
    scGPT/STATE/COMPASS/BulkFormer/BulkRNABert/CancerFoundation/UCE : FM embeddings (None if missing)
    response         : int (1=R, 0=NR)
    cohort           : str  cohort name
    cancer_type      : str  e.g. "SKCM"
    organ            : str  e.g. "Skin"
    ici_target       : str  e.g. "PD1"
    dataset          : "ICI"
"""

import argparse
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA as SklearnPCA

# Add repo root to path when running as a module
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing.preprocess_utils import (
    compute_max_mean_gene_set_features,
    create_sample_dict,
    ensure_genes_present,
    get_embedding_for_sample,
    load_foundation_model_embeddings,
    load_pca_embeddings,
    load_pickle,
    load_ssgsea_scores,
    get_pca_for_sample,
    pickle_data,
)

# ── Constants ──────────────────────────────────────────────────────────────────

CONFIG_PATH = REPO_ROOT / "configs" / "immunotherapy_cohorts.yaml"
TCGA_GENE_LIST_PATH = REPO_ROOT / "data" / "TCGA" / "TCGA_genes_ordered_by_variance.pkl"
OUTPUT_DIR = REPO_ROOT / "data" / "RNA_embeddings" / "ICI"

RESPONSE_MAP = {"R": 1, "NR": 0}   # response_label values in the h5ad obs


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_sample_ids(adata) -> list:
    """
    Return sample IDs from an ICI h5ad.
    Prefer a 'patient_id' obs column; fall back to obs_names.
    """
    if "patient_id" in adata.obs.columns:
        return adata.obs["patient_id"].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()


def find_ssgsea_intersection(cohorts_cfg: dict) -> dict:
    """
    Load ssGSEA DataFrames for every cohort and return the intersection of
    gene-set columns for each key (ssGSEA_AURORA, ssGSEA_immuno, etc.).
    """
    gs_file_map = {
        "ssGSEA_AURORA":    "aurora.pkl",
        "ssGSEA_immuno":    "immuno.pkl",
        "ssGSEA_hallmark":  "hallmark.pkl",
        "ssGSEA_combined":  "combined.pkl",
    }
    common_cols: dict = {}
    for gs_name, filename in gs_file_map.items():
        col_sets = []
        for cohort_name, cohort_cfg in cohorts_cfg.items():
            fp = REPO_ROOT / cohort_cfg["data_dir"].lstrip("./") / "ssGSEA_scores" / filename
            if fp.exists():
                df = load_pickle(fp)
                col_sets.append(set(df.columns))
        if col_sets:
            intersection = sorted(col_sets[0].intersection(*col_sets[1:]))
            common_cols[gs_name] = intersection
            print(f"  {gs_name}: intersection = {len(intersection)} gene sets "
                  f"(min per cohort = {min(len(c) for c in col_sets)}, "
                  f"max = {max(len(c) for c in col_sets)})")
    return common_cols


def build_cohort_pkl(cohort_name: str, cohort_cfg: dict, gene_names: dict,
                     ssgsea_common_cols: dict = None) -> dict:
    """
    Build the processed sample dictionary for one ICI cohort.

    Returns dict: { sample_id -> sample_dict }
    """
    data_dir = REPO_ROOT / cohort_cfg["data_dir"].lstrip("./")
    rna_path  = data_dir / "RNA" / f"{cohort_name}_bulk_RNA_processed_log1p.h5ad"

    print(f"\n{'='*70}")
    print(f"Processing cohort: {cohort_name}")
    print(f"  h5ad: {rna_path}")
    print(f"{'='*70}")

    # ── 1. Load expression ─────────────────────────────────────────────────
    adata = anndata.read_h5ad(rna_path)
    print(f"  Loaded: {adata.n_obs} samples x {adata.n_vars} genes")

    ici_arms = cohort_cfg.get("ici_arms")
    if ici_arms is not None:
        mask = adata.obs["ARM"].isin(ici_arms)
        n_excluded = (~mask).sum()
        adata = adata[mask].copy()
        print(f"  ARM filter ({ici_arms}): kept {adata.n_obs}, excluded {n_excluded}")

    sample_ids = get_sample_ids(adata)
    print(f"  Sample IDs (first 5): {sample_ids[:5]}")

    # ── 2. Build RNA-seq DataFrame, align to TCGA gene vocab ──────────────
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    rna_df = pd.DataFrame(X, index=sample_ids)
    # Assign gene symbols as columns
    if "gene_symbol" in adata.var.columns:
        rna_df.columns = adata.var["gene_symbol"].astype(str).tolist()
    else:
        rna_df.columns = adata.var_names.astype(str).tolist()

    # Collapse duplicate gene symbols (keep max)
    rna_df = rna_df.groupby(rna_df.columns, axis=1).max()
    # Align to TCGA vocab (adds zeros for missing genes, reorders)
    rna_df = ensure_genes_present(rna_df, gene_names)
    print(f"  RNA-seq after gene alignment: {rna_df.shape}")

    # ── 3. Gene-set features (max / mean pooling) ─────────────────────────
    gene_sets_folder = REPO_ROOT / "data" / "gene_sets" / "processed"
    print("  Computing gene-set features...")
    gene_set_features = compute_max_mean_gene_set_features(
        rna_df.copy(), list(gene_names.keys()), gene_sets_folder
    )

    # ── 4. ssGSEA scores ──────────────────────────────────────────────────
    ssgsea_folder = data_dir / "ssGSEA_scores"
    print(f"  Loading ssGSEA scores from {ssgsea_folder}...")
    ssgsea_scores = load_ssgsea_scores(ssgsea_folder, cohort_name)

    # ── 5. FM embeddings ──────────────────────────────────────────────────
    emb_folder = data_dir / "RNA" / "FM_embeddings"
    print(f"  Loading FM embeddings from {emb_folder}...")
    all_embeddings = load_foundation_model_embeddings(emb_folder, sample_ids)

    # ── 6. PCA embeddings ─────────────────────────────────────────────────
    pca_root = data_dir / "PCA"
    print(f"  Loading PCA embeddings from {pca_root}...")
    all_pca = load_pca_embeddings(pca_root, sample_ids, dims=[50, 100])
    print(f"    PCA_50:  {len(all_pca.get('PCA_50', {}))} samples")
    print(f"    PCA_100: {len(all_pca.get('PCA_100', {}))} samples")

    # ── 7. Response labels from obs ───────────────────────────────────────
    response_raw = adata.obs["response_label"].astype(str).tolist()
    response_labels = []
    for val in response_raw:
        mapped = RESPONSE_MAP.get(val.strip(), None)
        if mapped is None:
            print(f"  Warning: unknown response_label '{val}', skipping sample")
        response_labels.append(mapped)

    # ── 8. Assemble per-sample dicts ──────────────────────────────────────
    processed = {}
    n_skipped = 0

    for idx, sid in enumerate(sample_ids):
        # Skip samples with unknown response
        resp = response_labels[idx]
        if resp is None:
            n_skipped += 1
            continue

        # Gene-set features for this sample
        gs_features_sample = {k: v[idx] for k, v in gene_set_features.items()}

        # ssGSEA scores — reindex to intersection columns if provided
        obs_name = adata.obs_names[idx]
        for gs_name, scores_df in ssgsea_scores.items():
            cols = ssgsea_common_cols.get(gs_name) if ssgsea_common_cols else None
            if scores_df is not None and obs_name in scores_df.index:
                row = scores_df.loc[obs_name]
                if cols is not None:
                    row = row.reindex(cols, fill_value=0.0)
                gs_features_sample[gs_name] = row.to_numpy(dtype=np.float32)
            else:
                n_sets = len(cols) if cols is not None else (scores_df.shape[1] if scores_df is not None else 0)
                gs_features_sample[gs_name] = np.zeros(n_sets, dtype=np.float32)

        # FM embeddings
        embs = get_embedding_for_sample(sid, all_embeddings)

        # PCA embeddings
        pca_embs = get_pca_for_sample(sid, all_pca, dims=[50, 100])
        embs.update(pca_embs)

        processed[sid] = create_sample_dict(
            sample_id=sid,
            rna_seq=rna_df.iloc[idx].values.astype(np.float32),
            gene_set_features=gs_features_sample,
            embeddings=embs,
            dataset="ICI",
            cancer_type=cohort_cfg["cancer_type"],
            organ=cohort_cfg["organ"],
            # ICI-specific fields
            response=resp,
            cohort=cohort_name,
            ici_target=cohort_cfg["ici_target"],
        )

    print(f"\n  Built {len(processed)} samples  ({n_skipped} skipped — unknown response)")
    return processed


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build ICI cohort processed pickle files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cohort",
        type=str,
        help="Name of a single cohort to process (must match a key in immunotherapy_cohorts.yaml).",
    )
    group.add_argument(
        "--merge",
        action="store_true",
        help="Merge all per-cohort pickles listed in the config into ICI_all_processed.pkl.",
    )
    args = parser.parse_args()

    cfg = load_config()
    cohorts_cfg = cfg["cohorts"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        print("\n=== Merging all ICI cohort pickles ===")

        gene_names = load_pickle(TCGA_GENE_LIST_PATH)
        print(f"Loaded TCGA gene vocab: {len(gene_names)} genes")

        # ── 1. Find ssGSEA intersection columns across all cohorts ────────────
        print("\nFinding ssGSEA intersection columns...")
        ssgsea_common_cols = find_ssgsea_intersection(cohorts_cfg)

        # ── 2. Rebuild each cohort pkl with aligned ssGSEA ────────────────────
        merged = {}
        for cohort_name in cohorts_cfg:
            print(f"\nRebuilding {cohort_name} with aligned ssGSEA...")
            cohort_data = build_cohort_pkl(
                cohort_name, cohorts_cfg[cohort_name], gene_names,
                ssgsea_common_cols=ssgsea_common_cols,
            )
            out_path = OUTPUT_DIR / f"{cohort_name}_processed.pkl"
            pickle_data(cohort_data, out_path)
            print(f"  Saved {len(cohort_data)} samples → {out_path}")

            collision = set(merged.keys()) & set(cohort_data.keys())
            if collision:
                print(f"  WARNING: {len(collision)} ID collision(s). Consider prefixing IDs.")
            merged.update(cohort_data)

        print(f"\nMerged {len(merged)} samples total.")

        # ── 3. Recompute PCA on full merged RNA_seq ───────────────────────────
        print("\nRecomputing PCA on merged dataset...")
        sample_ids = list(merged.keys())
        rna_matrix = np.stack([merged[sid]["RNA_seq"] for sid in sample_ids])  # (N, 17490)
        print(f"  RNA matrix shape: {rna_matrix.shape}")

        for n_components in [50, 100]:
            key = f"PCA_{n_components}"
            print(f"  Fitting PCA({n_components})...")
            pca = SklearnPCA(n_components=n_components, random_state=42)
            projections = pca.fit_transform(rna_matrix).astype(np.float32)  # (N, n_components)
            for i, sid in enumerate(sample_ids):
                merged[sid][key] = projections[i]
            print(f"  {key}: done, shape={projections.shape}")

        # ── 4. Save ───────────────────────────────────────────────────────────
        out_path = OUTPUT_DIR / "ICI_all_processed.pkl"
        pickle_data(merged, out_path)
        print(f"\nSaved merged pkl ({len(merged)} samples) → {out_path}")
        return

    # Single cohort
    cohort_name = args.cohort
    if cohort_name not in cohorts_cfg:
        raise ValueError(
            f"Cohort '{cohort_name}' not found in {CONFIG_PATH}. "
            f"Available: {list(cohorts_cfg.keys())}"
        )

    gene_names = load_pickle(TCGA_GENE_LIST_PATH)
    print(f"Loaded TCGA gene vocab: {len(gene_names)} genes")

    processed = build_cohort_pkl(cohort_name, cohorts_cfg[cohort_name], gene_names)

    out_path = OUTPUT_DIR / f"{cohort_name}_processed.pkl"
    pickle_data(processed, out_path)
    print(f"\nSaved {len(processed)} samples → {out_path}")

    # Summary
    responses = [v["response"] for v in processed.values() if v.get("response") is not None]
    n_R  = sum(r == 1 for r in responses)
    n_NR = sum(r == 0 for r in responses)
    print(f"  Responders: {n_R}  Non-responders: {n_NR}")


if __name__ == "__main__":
    main()

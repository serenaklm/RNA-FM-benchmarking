"""Helpers shared by the foundation-model embedding scripts.

sample_ids_of: the ID each output embedding file is named by (same rule the scripts
    already used: aliquot IDs for TARGET_AML, patient barcodes for TCGA, `patient_id` where
    present, else obs_names).
restrict_to_samples: with --sample_ids, keep only the listed samples so that only their
    embeddings are (re)computed and overwritten. Used to re-embed a subset of a cohort, e.g.
    TCGA patients whose RNA-seq file changed, while leaving all other embeddings untouched.
"""


def sample_ids_of(adata, data_set):
    if data_set == "TARGET_AML":
        return adata.obs_names.astype(str).tolist()
    if data_set == "TCGA":
        return adata.obs["cases.submitter_id"].astype(str).tolist()
    if "patient_id" in adata.obs.columns:
        return adata.obs["patient_id"].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()  # ICI cohorts


def read_sample_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def restrict_to_samples(adata, data_set, sample_ids_path):
    """Return adata restricted to the IDs listed in sample_ids_path (one per line)."""
    if sample_ids_path is None:
        return adata
    wanted = set(read_sample_ids(sample_ids_path))
    ids = sample_ids_of(adata, data_set)
    missing = wanted - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} requested sample IDs not in input, e.g. {sorted(missing)[:3]}")
    mask = [i in wanted for i in ids]
    sub = adata[mask].copy()
    print(f"Restricted to {sub.n_obs} of {adata.n_obs} samples listed in {sample_ids_path}")
    return sub


def add_subset_args(parser, hvg=False):
    parser.add_argument(
        "--sample_ids", type=str, default=None,
        help="File with one sample ID per line; only these samples are embedded, and their "
             "existing embedding files are overwritten.")
    if hvg:
        parser.add_argument(
            "--hvg_reference", type=str, default=None,
            help="h5ad to run HVG selection on instead of the input (same genes, same order). "
                 "Used to keep a cohort's HVG list frozen when re-embedding a subset.")

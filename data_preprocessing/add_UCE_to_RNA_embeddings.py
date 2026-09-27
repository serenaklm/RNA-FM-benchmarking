"""
Adds a "UCE" key to every patient record in the consolidated RNA_embeddings
pickles, leaving everything else untouched.

Reads the pre-UCE pickles from data/RNA_embeddings_no_uce/ and writes the
augmented copies to data/RNA_embeddings/ (the path configs/all_models.yaml and
the ICI configs actually read from). Patients with no UCE embedding file get
None for that key (same convention as augment_TCGA_with_HVG_embeddings.py).

Does NOT touch data/RNA_embeddings_no_uce/ -- those are the backups.

Usage:
    python data_preprocessing/add_UCE_to_RNA_embeddings.py               # all datasets
    python data_preprocessing/add_UCE_to_RNA_embeddings.py --only CPTAC  # just CPTAC
"""
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR    = REPO_ROOT / "data" / "RNA_embeddings_no_uce"
OUT_DIR   = REPO_ROOT / "data" / "RNA_embeddings"
DATA_ROOT = REPO_ROOT / "data"

# filename (relative to IN_DIR / OUT_DIR) -> dataset name (used to locate
# data/{dataset}/RNA/FM_embeddings/UCE/{patient_id}.pkl)
PFS_DATASETS = {
    "TCGA_processed_cleaned.pkl": "TCGA",
    "CPTAC_processed.pkl": "CPTAC",
    "Hartwig_processed.pkl": "Hartwig",
    "TARGET_AML_processed.pkl": "TARGET_AML",
}

# ICI cohorts live under RNA_embeddings/ICI/ and their UCE embeddings under
# data/immunotherapy_cohorts/{cohort}/RNA/FM_embeddings/UCE/
ICI_DATASETS = {
    "ICI/Hugo_processed.pkl": "immunotherapy_cohorts/Hugo",
    "ICI/Riaz_processed.pkl": "immunotherapy_cohorts/Riaz",
    "ICI/Rose_processed.pkl": "immunotherapy_cohorts/Rose",
    "ICI/IMMotion150_processed.pkl": "immunotherapy_cohorts/IMMotion150",
    "ICI/IMVigor210_processed.pkl": "immunotherapy_cohorts/IMVigor210",
    "ICI/ICI_all_processed.pkl": None,  # combined file, handled separately below
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_emb(path: Path) -> np.ndarray:
    emb = load_pickle(path)
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


def index_uce_folder(uce_folder: Path) -> dict:
    if not uce_folder.exists():
        print(f"  WARNING: {uce_folder} not found -- all UCE keys will be None.")
        return {}
    index = {p.stem: p for p in uce_folder.glob("*.pkl")}
    print(f"  Indexed {len(index)} UCE embedding files from {uce_folder}")
    return index


def augment_file(in_path: Path, out_path: Path, uce_index: dict, label: str):
    print(f"\n=== {label} ===")
    print(f"Loading {in_path} ...")
    data = load_pickle(in_path)
    patient_ids = list(data.keys())
    print(f"Loaded {len(patient_ids)} patients.")

    missing = 0
    for patient_id in tqdm(patient_ids, desc=f"Augmenting {label}"):
        # The FM embedding files are keyed by the RNA-seq sample id, which for
        # some datasets (e.g. CPTAC) differs from the pkl's outer patient/case_id
        # key -- records from those datasets carry their own "sample_id" field
        # (set by the original get_*_embeddings.py builders) for exactly this
        # lookup. Falls back to patient_id for datasets where they're the same
        # (TCGA, Hartwig, TARGET_AML, and all ICI cohorts all have outer_key ==
        # sample_id already).
        lookup_id = data[patient_id].get("sample_id", patient_id)
        if lookup_id in uce_index:
            data[patient_id]["UCE"] = load_emb(uce_index[lookup_id])
        else:
            data[patient_id]["UCE"] = None
            missing += 1

    print(f"  Missing UCE embedding for {missing} / {len(patient_ids)} patients.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path} ...")
    pickle_data(data, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated subset of dataset/cohort names to process "
             "(e.g. 'CPTAC' or 'Hugo,Riaz'). Also accepts 'ICI_all' for the "
             "combined ICI file. Omit to process everything. Use this to "
             "rewrite just one file without touching pickles that other "
             "running jobs may currently have open.",
    )
    args = parser.parse_args()
    only = set(x.strip() for x in args.only.split(",")) if args.only else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- PFS datasets (TCGA, CPTAC, Hartwig, TARGET_AML) ----
    for filename, dataset in PFS_DATASETS.items():
        if only is not None and dataset not in only:
            continue
        uce_folder = DATA_ROOT / dataset / "RNA" / "FM_embeddings" / "UCE"
        uce_index = index_uce_folder(uce_folder)
        augment_file(IN_DIR / filename, OUT_DIR / filename, uce_index, dataset)

    # ---- ICI cohorts (per-cohort files) ----
    per_cohort_index = {}
    for filename, cohort_path in ICI_DATASETS.items():
        if cohort_path is None:
            continue
        cohort_name = cohort_path.split("/")[-1]
        uce_folder = DATA_ROOT / cohort_path / "RNA" / "FM_embeddings" / "UCE"
        uce_index = index_uce_folder(uce_folder)
        per_cohort_index[filename] = uce_index
        if only is not None and cohort_name not in only:
            continue
        augment_file(IN_DIR / filename, OUT_DIR / filename, uce_index, cohort_name)

    # ---- ICI combined file: union of all per-cohort UCE indices ----
    if only is None or "ICI_all" in only:
        combined_index = {}
        for idx in per_cohort_index.values():
            combined_index.update(idx)
        augment_file(
            IN_DIR / "ICI/ICI_all_processed.pkl",
            OUT_DIR / "ICI/ICI_all_processed.pkl",
            combined_index,
            "ICI_all (combined)",
        )

    print("\nDone. Originals in data/RNA_embeddings_no_uce/ were not modified.")


if __name__ == "__main__":
    main()

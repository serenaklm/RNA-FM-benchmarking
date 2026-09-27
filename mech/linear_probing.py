"""
linear_probing.py

Trains bias-free linear probes (nn.Linear, no bias) with cross-entropy to predict
clinical labels from RNA embeddings across 4 datasets x 32 embeddings x 3 tasks x 5 splits.

Tasks:
  cancer_type  — TCGA, CPTAC, Hartwig   — accuracy + macro F1
  grade        — TCGA, CPTAC            — accuracy + macro F1 + Cohen's kappa
                 (two-tier: low = G1/G2/Low Grade, high = G3/G4/High Grade; GX/GB excluded)
  organ        — TCGA, CPTAC, Hartwig   — accuracy + macro F1  (primary tumor site)

Usage:
  python -m mech.linear_probing
  python -m mech.linear_probing --out_csv /path/to/results.csv --device cuda:1
"""

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

BASE = "/data/rbg/users/ashyamal/RNA-FM-benchmarking"

DATASETS = ["TCGA", "CPTAC", "Hartwig", "TARGET_AML"]

# Tasks and which datasets they apply to.
# TARGET_AML: single cancer type + single organ (trivial) → skipped.
# Hartwig: metastatic, no AJCC staging → stage skipped.
TASK_DATASETS = {
    "cancer_type": ["TCGA", "CPTAC", "Hartwig"],
    "grade":       ["TCGA", "CPTAC"],
    "organ":       ["TCGA", "CPTAC", "Hartwig"],
}

# 30 embedding specs: (rna_key, n_hvgs)
# For RNA_seq, n_hvgs slices the first n genes at runtime.
# For all other keys, n_hvgs is None and the full embedding is used.
EMBEDDING_SPECS = [
    # Bulk FMs
    ("BulkFormer",       None),
    ("BulkRNABert",      None),
    ("COMPASS",          None),
    # Single-cell FMs
    ("scGPT",            None),
    ("STATE",            None),
    ("CancerFoundation", None),
    ("UCE",              None),
    # ssGSEA gene sets
    ("ssGSEA_AURORA",    None),
    ("ssGSEA_immuno",    None),
    ("ssGSEA_hallmark",  None),
    ("ssGSEA_combined",  None),
    # Max gene sets
    ("max_AURORA",       None),
    ("max_immuno",       None),
    ("max_hallmark",     None),
    ("max_combined",     None),
    # PCA
    ("PCA_50",           None),
    ("PCA_100",          None),
    # HVG slices of RNA_seq
    ("RNA_seq",          18),
    ("RNA_seq",          50),
    ("RNA_seq",          132),
    ("RNA_seq",          500),
    ("RNA_seq",          729),
    ("RNA_seq",          1250),
    ("RNA_seq",          5000),
    ("RNA_seq",          10000),
    # HVG4 — matches FM embedding dimensionalities
    ("RNA_seq",          44),
    ("RNA_seq",          256),
    ("RNA_seq",          512),
    ("RNA_seq",          640),
    ("RNA_seq",          1280),
    ("RNA_seq",          2058),
    # Full raw expression
    ("RNA_seq",          None),
]

COMPASS_MAPPING_PATHS = {
    "CPTAC":   f"{BASE}/data/CPTAC/id_to_tcga_mapping.pkl",
    "Hartwig": f"{BASE}/data/Hartwig/id_to_tcga_mapping.pkl",
}

BAD_VALUES = {"unknown", "not reported", "'--", "not applicable", "n/a", "na", "none", "", "nan/null", "--"}

# Two-tier tumor grade. GDC tumor_grade mixes G1–G4, a two-tier High/Low Grade
# system (e.g. BLCA), and non-ordinal codes (GX = cannot be assessed,
# GB = borderline); the latter are excluded.
GRADE_TWO_TIER = {
    "G1": "low",  "G2": "low",  "Low Grade":  "low",
    "G3": "high", "G4": "high", "High Grade": "high",
}


# ── Data loading (mirrors train_simple_clinical_baseline.py) ──────────────────

def clean_clinical(df):
    def _clean(x):
        if pd.isnull(x):
            return np.nan
        if str(x).strip().lower() in BAD_VALUES:
            return np.nan
        return x
    return df.apply(lambda col: col.map(_clean))


def dedup_gdc_clinical(clinical, id_col):
    """
    Collapse a GDC clinical table (one row per diagnosis/treatment record) to one
    row per patient. Must be called after clean_clinical.
      - Non-diagnosis fields (demographics, project): first non-null value across rows.
      - diagnoses.* fields: first non-null value among primary-disease diagnosis rows
        only, so stage/grade never come from a recurrence or prior/other primary.
    """
    out = clinical.groupby(id_col).first()
    flag = "diagnoses.diagnosis_is_primary_disease"
    if flag in clinical.columns:
        diag_cols = [c for c in clinical.columns if c.startswith("diagnoses.")]
        is_primary = clinical[flag].astype(str).str.strip().str.lower() == "true"   # TCGA "True", CPTAC "true"
        prim = clinical[is_primary].groupby(id_col)[diag_cols].first()
        out[diag_cols] = prim.reindex(out.index)
    return out


def load_dataset(dataset_name):
    """Returns (patient_data, clinical_dedup, split_dir, sample_to_patient)."""
    if dataset_name == "TCGA":
        with open(f"{BASE}/data/RNA_embeddings/TCGA_processed_cleaned.pkl", "rb") as f:
            patient_data = pickle.load(f)
        clinical = clean_clinical(pd.read_csv(
            f"{BASE}/data/TCGA/clinical/clinical.tsv", sep="\t", low_memory=False
        ))
        clinical_dedup = dedup_gdc_clinical(clinical, "cases.submitter_id")
        return patient_data, clinical_dedup, f"{BASE}/splits/TCGA_cleaned", None

    elif dataset_name == "CPTAC":
        with open(f"{BASE}/data/RNA_embeddings/CPTAC_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        clinical = clean_clinical(pd.read_csv(
            "/data/rbg/shared/datasets/CPTAC/clinical/clinical.tsv", sep="\t", low_memory=False
        ))
        clinical_dedup = dedup_gdc_clinical(clinical, "cases.case_id")
        return patient_data, clinical_dedup, f"{BASE}/splits/CPTAC", None

    elif dataset_name == "Hartwig":
        with open(f"{BASE}/data/RNA_embeddings/Hartwig_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        with open(f"{BASE}/data/Hartwig/sample_to_patient_mapping.pkl", "rb") as f:
            sample_to_patient = pickle.load(f)
        metadata = pd.read_csv(
            "/data/rbg/shared/datasets/Hartwig/isofox_tpm_update_4/metadata.tsv", sep="\t"
        )
        clinical_dedup = metadata.drop_duplicates(subset="sampleId").set_index("sampleId")
        return patient_data, clinical_dedup, f"{BASE}/splits/Hartwig", sample_to_patient

    elif dataset_name == "TARGET_AML":
        with open(f"{BASE}/data/RNA_embeddings/TARGET_AML_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        clinical = clean_clinical(pd.read_csv(
            f"{BASE}/data/TARGET_AML/clinical/clinical.tsv", sep="\t", low_memory=False
        ))
        clinical_dedup = dedup_gdc_clinical(clinical, "cases.submitter_id")
        return patient_data, clinical_dedup, f"{BASE}/splits/TARGET_AML", None

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_clinical_id(sid, dataset_name):
    if dataset_name == "TARGET_AML":
        return "-".join(sid.rsplit("-", 1)[:1])
    return sid


def deduplicate_by_patient(sample_ids, s2p):
    seen, result = set(), []
    for sid in sample_ids:
        pid = s2p.get(sid, sid) if s2p else sid
        if pid not in seen:
            seen.add(pid)
            result.append(sid)
    return result


# ── Label + embedding extraction ──────────────────────────────────────────────

def get_label(task, rec, clinical_row):
    if task == "cancer_type":
        val = rec.get("cancer_type")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        val = str(val).strip()
        return None if val.lower() in BAD_VALUES else val

    elif task == "organ":
        val = rec.get("organ")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        val = str(val).strip()
        return None if val.lower() in BAD_VALUES else val

    elif task == "grade":
        if clinical_row is None:
            return None
        val = clinical_row.get("diagnoses.tumor_grade", np.nan)
        if pd.isnull(val):
            return None
        return GRADE_TWO_TIER.get(str(val).strip())

    return None


def get_embedding(rec, rna_key, n_hvgs):
    x = rec.get(rna_key)
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1 or len(x) == 0:
        return None
    if rna_key == "RNA_seq" and n_hvgs is not None:
        x = x[:n_hvgs]
    if not np.isfinite(x).all():
        return None
    return x


# ── Linear probe ──────────────────────────────────────────────────────────────

def train_linear_probe(X_train, y_train, X_val, y_val, n_classes, device,
                       lr=1e-3, max_epochs=500, patience=50):
    model = nn.Linear(X_train.shape[1], n_classes, bias=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_train).to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_vl = torch.from_numpy(X_val).to(device)
    y_vl = torch.from_numpy(y_val).long().to(device)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        criterion(model(X_tr), y_tr).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), y_vl).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(task, y_true, y_pred):
    metrics = {
        "accuracy":       float(np.mean(y_true == y_pred)),
        "macro_f1":       float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa":          float("nan"),
    }
    if task == "grade":
        try:
            # grade is binary (two-tier), so unweighted Cohen's kappa
            metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            pass
    return metrics


# ── Per-split runner ──────────────────────────────────────────────────────────

def collect_split(split_ids, patient_data, clinical_dedup, dataset_name,
                  rna_key, n_hvgs, task, s2p, compass_valid_ids):
    X, y = [], []
    for sid in split_ids:
        if sid not in patient_data:
            continue
        if compass_valid_ids is not None and sid not in compass_valid_ids:
            continue
        rec = patient_data[sid]
        emb = get_embedding(rec, rna_key, n_hvgs)
        if emb is None:
            continue
        cid = get_clinical_id(sid, dataset_name)
        crow = clinical_dedup.loc[cid] if cid in clinical_dedup.index else None
        label = get_label(task, rec, crow)
        if label is None:
            continue
        X.append(emb)
        y.append(label)
    return X, y


def run_one_split(dataset, rna_key, n_hvgs, task, split_idx,
                  patient_data, clinical_dedup, split_dir, s2p,
                  compass_valid_ids, device):
    split_path = Path(split_dir) / f"split_{split_idx}.pkl"
    if not split_path.exists():
        return None

    with open(split_path, "rb") as f:
        split = pickle.load(f)

    all_ids = split["train"] + split.get("val", []) + split["test"]
    s2p_local = (
        {s: "-".join(s.rsplit("-", 1)[:1]) for s in all_ids}
        if dataset == "TARGET_AML" else s2p
    )

    train_ids = deduplicate_by_patient(split["train"],        s2p_local)
    val_ids   = deduplicate_by_patient(split.get("val", []), s2p_local)
    test_ids  = deduplicate_by_patient(split["test"],         s2p_local)

    # For PCA and HVG, collect full RNA_seq and apply train-only transforms below.
    is_pca = rna_key.startswith("PCA_")
    needs_hvg_transform = (rna_key == "RNA_seq" and n_hvgs is not None)
    collect_key    = "RNA_seq" if (is_pca or needs_hvg_transform) else rna_key
    collect_n_hvgs = None      if (is_pca or needs_hvg_transform) else n_hvgs

    ckwargs = dict(
        patient_data=patient_data, clinical_dedup=clinical_dedup,
        dataset_name=dataset, rna_key=collect_key, n_hvgs=collect_n_hvgs,
        task=task, s2p=s2p_local, compass_valid_ids=compass_valid_ids,
    )

    X_tr_raw, y_tr_raw = collect_split(train_ids, **ckwargs)
    X_vl_raw, y_vl_raw = collect_split(val_ids,   **ckwargs)
    X_te_raw, y_te_raw = collect_split(test_ids,  **ckwargs)

    if len(X_tr_raw) == 0 or len(X_te_raw) == 0:
        return None

    # Fit PCA or HVG selection on training data only, then apply to all splits.
    if is_pca or needs_hvg_transform:
        train_mat = np.stack(X_tr_raw).astype(np.float32)
        if is_pca:
            n_components = int(rna_key.split("_")[1])
            n_actual = min(n_components, train_mat.shape[0] - 1, train_mat.shape[1])
            pca = PCA(n_components=n_actual, random_state=0)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api="blas"):
                pca.fit(train_mat)
            X_tr_raw = list(pca.transform(train_mat))
            if X_vl_raw:
                X_vl_raw = list(pca.transform(np.stack(X_vl_raw).astype(np.float32)))
            if X_te_raw:
                X_te_raw = list(pca.transform(np.stack(X_te_raw).astype(np.float32)))
        else:  # HVG
            train_var = np.var(train_mat, axis=0)
            hvg_idx = np.argsort(train_var)[::-1][:n_hvgs]
            X_tr_raw = [x[hvg_idx] for x in X_tr_raw]
            X_vl_raw = [x[hvg_idx] for x in X_vl_raw]
            X_te_raw = [x[hvg_idx] for x in X_te_raw]

    le = LabelEncoder().fit(y_tr_raw)
    if len(le.classes_) < 2:
        return None  # trivial single-class task

    known = set(le.classes_)

    def encode(X_raw, y_raw):
        pairs = [(x, lbl) for x, lbl in zip(X_raw, y_raw) if lbl in known]
        if not pairs:
            return None, None
        Xf, yf = zip(*pairs)
        return np.stack(Xf).astype(np.float32), le.transform(list(yf)).astype(np.int64)

    X_train = np.stack(X_tr_raw).astype(np.float32)
    y_train = le.transform(y_tr_raw).astype(np.int64)
    X_val,  y_val  = encode(X_vl_raw, y_vl_raw)
    X_test, y_test = encode(X_te_raw, y_te_raw)

    if X_test is None:
        return None

    # Fall back to train for early stopping if val is empty
    if X_val is None:
        X_val, y_val = X_train, y_train

    model = train_linear_probe(X_train, y_train, X_val, y_val,
                               n_classes=len(le.classes_), device=device)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test).to(device)).argmax(dim=1).cpu().numpy()

    return {
        "dataset":   dataset,
        "rna_key":   rna_key,
        "n_hvgs":    n_hvgs,
        "task":      task,
        "split":     split_idx,
        "n_train":   len(X_train),
        "n_val":     len(X_val),
        "n_test":    len(X_test),
        "n_classes": len(le.classes_),
        **compute_metrics(task, y_test, y_pred),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Restrict to a single dataset and/or embedding range if specified
    datasets  = [args.dataset] if args.dataset else DATASETS
    emb_specs = EMBEDDING_SPECS[args.emb_start:args.emb_end]

    # Load existing results for resuming
    done_keys = set()
    if out_csv.exists() and not args.overwrite:
        existing = pd.read_csv(out_csv)
        if "n_hvgs" not in existing.columns:
            existing["n_hvgs"] = None
        for _, row in existing.iterrows():
            done_keys.add((
                row["dataset"], row["rna_key"],
                None if pd.isna(row["n_hvgs"]) else int(row["n_hvgs"]),
                row["task"], int(row["split"]),
            ))
        print(f"Loaded {len(done_keys)} already-done runs from {out_csv}")

    rows = []

    for dataset in datasets:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")

        tasks_for_ds = [t for t, ds_list in TASK_DATASETS.items() if dataset in ds_list]
        if not tasks_for_ds:
            print(f"  No applicable tasks for {dataset}, skipping.")
            continue

        patient_data, clinical_dedup, split_dir, s2p = load_dataset(dataset)

        # Load COMPASS valid IDs once per dataset
        compass_valid_ids = None
        if dataset in COMPASS_MAPPING_PATHS:
            with open(COMPASS_MAPPING_PATHS[dataset], "rb") as f:
                compass_valid_ids = set(pickle.load(f).keys())

        total = len(emb_specs) * len(tasks_for_ds) * args.n_splits
        pbar = tqdm(total=total, desc=dataset)

        for rna_key, n_hvgs in emb_specs:
            cv_ids = compass_valid_ids if rna_key == "COMPASS" else None

            for task in tasks_for_ds:
                for split_idx in range(args.n_splits):
                    pbar.update(1)

                    key = (dataset, rna_key,
                           None if n_hvgs is None else int(n_hvgs),
                           task, split_idx)
                    if key in done_keys:
                        continue

                    try:
                        result = run_one_split(
                            dataset=dataset, rna_key=rna_key, n_hvgs=n_hvgs,
                            task=task, split_idx=split_idx,
                            patient_data=patient_data, clinical_dedup=clinical_dedup,
                            split_dir=split_dir, s2p=s2p,
                            compass_valid_ids=cv_ids, device=device,
                        )
                    except Exception as e:
                        print(f"\n  ERROR {dataset} {rna_key} {task} split={split_idx}: {e}")
                        result = {
                            "dataset": dataset, "rna_key": rna_key, "n_hvgs": n_hvgs,
                            "task": task, "split": split_idx,
                            "n_train": None, "n_val": None, "n_test": None, "n_classes": None,
                            "accuracy": None, "macro_f1": None, "kappa": None,
                            "error": str(e),
                        }

                    if result is not None:
                        rows.append(result)

        pbar.close()

        # Write incrementally after each dataset so progress is never lost
        if rows:
            df_new = pd.DataFrame(rows)
            if out_csv.exists() and not args.overwrite:
                df_new = pd.concat([pd.read_csv(out_csv), df_new], ignore_index=True, sort=False)
            df_new.to_csv(out_csv, index=False)
            print(f"  Saved {len(df_new)} rows to {out_csv}")
            rows = []

    print(f"\nDone. Results at: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", type=str,
                        default=f"{BASE}/benchmarking_results/linear_probing_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing results and rerun everything")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"],
                        help="Run a single dataset (default: all)")
    parser.add_argument("--emb_start", type=int, default=0,
                        help="Start index into EMBEDDING_SPECS (inclusive)")
    parser.add_argument("--emb_end", type=int, default=len(EMBEDDING_SPECS),
                        help="End index into EMBEDDING_SPECS (exclusive)")
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""
train_clinical_baseline.py

Extended Cox Ridge clinical baseline using all sufficiently-available clinical
features per dataset.  Unlike train_simple_clinical_baseline.py, which only
uses age / gender / stage, this script encodes the full feature set and
extracts per-feature Cox coefficients so features can be ranked by
predictive value (useful for UMAP coloring).

Encoding strategy
-----------------
  continuous : z-score (train median/std), missing → 0 + binary __missing flag
  ordinal    : integer map, missing → 0 + binary __missing flag
  binary     : yes/no/true/false → {0,1}, missing → 0.5 + binary __missing flag
  categorical: one-hot with explicit MISSING category; rare cats (< min_cat_freq) → OTHER

This means no patients are dropped for missing clinical data; missingness is
modelled explicitly.
"""
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


# ---------------------------------------------------------------------------
# Constants & ordinal maps
# ---------------------------------------------------------------------------
BAD_VALUES = {"unknown", "not reported", "'--", "not applicable", "n/a", "na",
              "none", "", "nan/null", "--"}

DATASETS = ["TCGA", "Hartwig", "CPTAC", "TARGET_AML"]

STAGE_MAP = {
    'Stage 0': 0, 'Stage 0a': 0, 'Stage 0is': 0, 'Stage IS': 0,
    'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage IB1': 1, 'Stage IC': 1,
    'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage IIC': 2,
    'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
    'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4, 'Stage IVC': 4,
}
T_MAP = {
    'T0': 0, 'Tis': 0,
    'T1': 1, 'T1a': 1, 'T1b': 1, 'T1c': 1, 'T1d': 1, 'T1mi': 1,
    'T2': 2, 'T2a': 2, 'T2b': 2, 'T2c': 2,
    'T3': 3, 'T3a': 3, 'T3b': 3, 'T3c': 3,
    'T4': 4, 'T4a': 4, 'T4b': 4, 'T4c': 4, 'T4d': 4,
}
N_MAP = {
    'N0': 0, 'N0 (i-)': 0, 'N0 (i+)': 0, 'N0 (mol-)': 0, 'N0 (mol+)': 0,
    'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1, 'N1mi': 1,
    'N2': 2, 'N2a': 2, 'N2b': 2, 'N2c': 2,
    'N3': 3, 'N3a': 3, 'N3b': 3, 'N3c': 3,
}
M_MAP = {
    'M0': 0, 'cM0 (i+)': 0,
    'M1': 1, 'M1a': 1, 'M1b': 1, 'M1c': 1, 'M1d': 1,
}
BOOL_MAP = {
    'yes': 1.0, 'no': 0.0,
    'true': 1.0, 'false': 0.0,
    '1': 1.0, '0': 0.0,
    'present': 1.0, 'absent': 0.0,
}

# Feature spec per dataset: list of (type, column_name [, mapping])
#   "continuous"  — real-valued; z-scored
#   "ordinal"     — integer map supplied as third element
#   "binary"      — yes/no, true/false → {0,1}
#   "categorical" — one-hot encoding
FEATURE_SPECS = {
    "TCGA": [
        ("categorical", "project.project_id"),
        ("continuous",  "demographic.age_at_index"),
        ("categorical", "demographic.gender"),
        ("categorical", "demographic.race"),
        ("ordinal",     "diagnoses.ajcc_pathologic_stage",      STAGE_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_t",          T_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_n",          N_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_m",          M_MAP),
        ("binary",      "diagnoses.prior_malignancy"),
        ("binary",      "diagnoses.prior_treatment"),
        ("binary",      "diagnoses.synchronous_malignancy"),
        ("categorical", "diagnoses.tissue_or_organ_of_origin"),
        ("categorical", "diagnoses.tumor_grade"),
        ("categorical", "diagnoses.residual_disease"),
        ("categorical", "diagnoses.morphology"),
        ("categorical", "diagnoses.primary_diagnosis"),
    ],
    "CPTAC": [
        ("continuous",  "age_years"),           # derived in load_dataset
        ("categorical", "demographic.gender"),
        ("categorical", "demographic.race"),
        ("ordinal",     "diagnoses.ajcc_pathologic_stage",      STAGE_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_t",          T_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_n",          N_MAP),
        ("ordinal",     "diagnoses.ajcc_pathologic_m",          M_MAP),
        ("categorical", "diagnoses.tissue_or_organ_of_origin"),
        ("categorical", "diagnoses.tumor_grade"),
        ("categorical", "diagnoses.residual_disease"),
        ("categorical", "diagnoses.morphology"),
        ("categorical", "diagnoses.primary_diagnosis"),
    ],
    "TARGET_AML": [
        ("continuous",  "demographic.age_at_index"),
        ("categorical", "demographic.gender"),
        ("categorical", "demographic.race"),
        ("categorical", "demographic.ethnicity"),
    ],
    "Hartwig": [
        ("continuous",  "age_at_biopsy"),       # derived in load_dataset
        ("categorical", "gender"),
        ("categorical", "primaryTumorLocation"),
        ("binary",      "hasSystemicPreTreatment"),
        ("binary",      "hasRadiotherapyPreTreatment"),
    ],
}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def clean_val(x):
    """Return np.nan for bad/missing strings; otherwise return the value."""
    if pd.isnull(x):
        return np.nan
    return np.nan if str(x).strip().lower() in BAD_VALUES else x


def clean_df(df):
    return df.apply(lambda col: col.map(clean_val))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(dataset_name, base):
    if dataset_name == "TCGA":
        with open(f"{base}/data/RNA_embeddings/TCGA_processed_cleaned.pkl", "rb") as f:
            patient_data = pickle.load(f)
        clinical = clean_df(pd.read_csv(
            f"{base}/data/TCGA/clinical/clinical.tsv", sep="\t", low_memory=False))
        clinical_dedup = (clinical.drop_duplicates(subset="cases.submitter_id")
                          .set_index("cases.submitter_id"))
        return patient_data, clinical_dedup, f"{base}/splits/TCGA_cleaned", None

    elif dataset_name == "CPTAC":
        with open(f"{base}/data/RNA_embeddings/CPTAC_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        clinical = clean_df(pd.read_csv(
            "/data/rbg/shared/datasets/CPTAC/clinical/clinical.tsv", sep="\t", low_memory=False))
        clinical_dedup = (clinical.drop_duplicates(subset="cases.case_id")
                          .set_index("cases.case_id"))
        # age_at_diagnosis is in days in GDC format
        clinical_dedup["age_years"] = (
            pd.to_numeric(clinical_dedup.get("diagnoses.age_at_diagnosis"), errors="coerce")
            / 365.25
        )
        return patient_data, clinical_dedup, f"{base}/splits/CPTAC", None

    elif dataset_name == "Hartwig":
        with open(f"{base}/data/RNA_embeddings/Hartwig_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        with open(f"{base}/data/Hartwig/sample_to_patient_mapping.pkl", "rb") as f:
            sample_to_patient = pickle.load(f)
        metadata = pd.read_csv(
            "/data/rbg/shared/datasets/Hartwig/isofox_tpm_update_4/metadata.tsv", sep="\t")
        metadata["biopsyDate"]   = pd.to_datetime(metadata["biopsyDate"])
        metadata["age_at_biopsy"] = metadata["biopsyDate"].dt.year - metadata["birthYear"]
        clinical_dedup = (metadata.drop_duplicates(subset="sampleId")
                          .set_index("sampleId"))
        return patient_data, clinical_dedup, f"{base}/splits/Hartwig", sample_to_patient

    elif dataset_name == "TARGET_AML":
        with open(f"{base}/data/RNA_embeddings/TARGET_AML_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        clinical = clean_df(pd.read_csv(
            f"{base}/data/TARGET_AML/clinical/clinical.tsv", sep="\t", low_memory=False))
        clinical_dedup = (clinical.drop_duplicates(subset="cases.submitter_id")
                          .set_index("cases.submitter_id"))
        return patient_data, clinical_dedup, f"{base}/splits/TARGET_AML", None

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# ID helpers  (unchanged from train_simple_clinical_baseline.py)
# ---------------------------------------------------------------------------
def deduplicate_by_patient(sample_ids, s2p):
    seen, result = set(), []
    for sid in sample_ids:
        pid = s2p.get(sid, sid) if s2p else sid
        if pid not in seen:
            seen.add(pid)
            result.append(sid)
    return result


def get_clinical_id(sid, dataset_name):
    if dataset_name == "TARGET_AML":
        return "-".join(sid.rsplit("-", 1)[:1])
    return sid


def get_valid_ids(dataset_name, patient_data, clinical_dedup, ids):
    """Keep IDs that have a patient_data entry, a clinical row, and valid PFS."""
    valid = []
    for sid in ids:
        if sid not in patient_data:
            continue
        cid = get_clinical_id(sid, dataset_name)
        if cid not in clinical_dedup.index:
            continue
        pfs_time  = patient_data[sid].get("PFS_time",  None)
        pfs_event = patient_data[sid].get("PFS_event", None)
        if pfs_time is None or pfs_event is None or float(pfs_time) < 0:
            continue
        valid.append(sid)
    return valid


# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------
class FeatureEncoder:
    """
    Fits on training data and transforms train/test consistently.
    Feature names are stored in self.feature_names_ after fit_transform().
    """

    def __init__(self, min_cat_freq=10):
        self.min_cat_freq = min_cat_freq
        self._specs = []
        self.params_ = {}
        self.feature_names_ = []

    def set_specs(self, specs):
        self._specs = specs

    # ------------------------------------------------------------------ fit
    def fit_transform(self, df):
        self.params_ = {}
        self.feature_names_ = []
        return self._encode(df, fitting=True)

    def transform(self, df):
        return self._encode(df, fitting=False)

    # ---------------------------------------------------------------- encode
    def _encode(self, df, fitting):
        parts = []
        for spec in self._specs:
            ftype, col_name = spec[0], spec[1]
            col = (df[col_name] if col_name in df.columns
                   else pd.Series([np.nan] * len(df), name=col_name, index=df.index))
            if ftype == "continuous":
                parts.append(self._enc_continuous(col_name, col, fitting))
            elif ftype == "ordinal":
                parts.append(self._enc_ordinal(col_name, col, spec[2], fitting))
            elif ftype == "binary":
                parts.append(self._enc_binary(col_name, col, fitting))
            elif ftype == "categorical":
                parts.append(self._enc_categorical(col_name, col, fitting))
        return np.hstack(parts) if parts else np.zeros((len(df), 0))

    # ---------------------------------------------------- per-type encoders
    def _enc_continuous(self, name, col, fitting):
        vals = pd.to_numeric(col.map(clean_val), errors="coerce")
        if fitting:
            med = float(np.nanmedian(vals))
            std = float(np.nanstd(vals))
            self.params_[name] = {"median": med, "std": std if std > 0 else 1.0}
            self.feature_names_ += [name, f"{name}__missing"]
        p       = self.params_[name]
        missing = vals.isna().astype(float).values
        scaled  = ((vals.fillna(p["median"]) - p["median"]) / p["std"]).values
        return np.column_stack([scaled, missing])

    def _enc_ordinal(self, name, col, mapping, fitting):
        cleaned = col.map(clean_val)
        mapped  = cleaned.map(
            lambda x: mapping.get(str(x), np.nan) if not pd.isnull(x) else np.nan)
        if fitting:
            self.params_[name] = {}
            self.feature_names_ += [name, f"{name}__missing"]
        missing = mapped.isna().astype(float).values
        imputed = mapped.fillna(0).values.astype(float)
        return np.column_stack([imputed, missing])

    def _enc_binary(self, name, col, fitting):
        cleaned = col.map(clean_val)
        mapped  = cleaned.map(
            lambda x: BOOL_MAP.get(str(x).strip().lower(), np.nan)
            if not pd.isnull(x) else np.nan)
        if fitting:
            self.params_[name] = {}
            self.feature_names_ += [name, f"{name}__missing"]
        missing = mapped.isna().astype(float).values
        imputed = mapped.fillna(0.5).values.astype(float)   # 0.5 = unknown
        return np.column_stack([imputed, missing])

    def _enc_categorical(self, name, col, fitting):
        cleaned = col.map(clean_val).fillna("MISSING").astype(str).str.strip()
        if fitting:
            counts = cleaned.value_counts()
            keep   = set(counts[counts >= self.min_cat_freq].index) | {"MISSING"}
            cats   = sorted(keep)
            self.params_[name] = {"cats": cats}
            self.feature_names_ += [f"{name}__{c}" for c in cats]
        cats    = self.params_[name]["cats"]
        cat_idx = {c: i for i, c in enumerate(cats)}
        mapped  = cleaned.map(lambda x: x if x in cat_idx else "MISSING")
        result  = np.zeros((len(col), len(cats)), dtype=float)
        for j, val in enumerate(mapped):
            result[j, cat_idx[val]] = 1.0
        return result


# ---------------------------------------------------------------------------
# Build (X, y)
# ---------------------------------------------------------------------------
def prepare_Xy(dataset_name, patient_data, clinical_dedup, ids, encoder, fitting=False):
    rows = [clinical_dedup.loc[get_clinical_id(sid, dataset_name)].to_dict()
            for sid in ids]
    df = pd.DataFrame(rows, index=ids)

    X = encoder.fit_transform(df) if fitting else encoder.transform(df)
    y = np.array(
        [(bool(patient_data[sid]["PFS_event"]), float(patient_data[sid]["PFS_time"]))
         for sid in ids],
        dtype=[("event", bool), ("time", float)],
    )
    return X, y


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_dataset(dataset_name, n_splits, alpha, base, min_cat_freq):
    patient_data, clinical_dedup, split_dir, sample_to_patient = \
        load_dataset(dataset_name, base)
    split_dir = Path(split_dir)
    specs     = FEATURE_SPECS[dataset_name]

    cindex_per_split = []
    coef_per_split   = []   # list of (feature_names list, coef ndarray)

    for i in range(n_splits):
        split_path = split_dir / f"split_{i}.pkl"
        if not split_path.exists():
            print(f"[{dataset_name}] WARNING: {split_path} missing, skipping")
            continue

        split = pickle.load(open(split_path, "rb"))

        if dataset_name == "TARGET_AML":
            s2p = {s: "-".join(s.rsplit("-", 1)[:1]) for s in split["train"] + split["test"]}
        elif dataset_name == "Hartwig":
            s2p = sample_to_patient
        else:
            s2p = None

        raw_train = deduplicate_by_patient(split["train"], s2p)
        raw_test  = deduplicate_by_patient(split["test"],  s2p)

        train_ids = get_valid_ids(dataset_name, patient_data, clinical_dedup, raw_train)
        test_ids  = get_valid_ids(dataset_name, patient_data, clinical_dedup, raw_test)

        if not train_ids or not test_ids:
            print(f"[{dataset_name}] Split {i}: empty after filtering, skipping")
            continue

        encoder = FeatureEncoder(min_cat_freq=min_cat_freq)
        encoder.set_specs(specs)

        X_train, y_train = prepare_Xy(dataset_name, patient_data, clinical_dedup,
                                      train_ids, encoder, fitting=True)
        X_test,  y_test  = prepare_Xy(dataset_name, patient_data, clinical_dedup,
                                      test_ids,  encoder, fitting=False)

        model = CoxPHSurvivalAnalysis(alpha=alpha)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"[{dataset_name}] Split {i}: model fit failed: {e}")
            continue

        ci = float(concordance_index_censored(
            y_test["event"], y_test["time"], model.predict(X_test))[0])
        cindex_per_split.append(ci)
        coef_per_split.append((list(encoder.feature_names_), model.coef_.copy()))

        print(f"[{dataset_name}] Split {i}: C-index={ci:.4f}  "
              f"n_train={len(X_train)}  n_test={len(X_test)}  "
              f"n_features={X_train.shape[1]}")

    if not cindex_per_split:
        return {"cindex_mean": np.nan, "cindex_std": np.nan}, pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------------
    # Aggregate coefficients across splits
    # Feature sets may differ slightly (rare categories), so align by name
    # ------------------------------------------------------------------
    all_names = sorted({n for names, _ in coef_per_split for n in names})
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    coef_matrix = np.zeros((len(coef_per_split), len(all_names)), dtype=float)
    for s_idx, (names, coefs) in enumerate(coef_per_split):
        for j, n in enumerate(names):
            coef_matrix[s_idx, name_to_idx[n]] = coefs[j]

    mean_coef = coef_matrix.mean(axis=0)
    std_coef  = coef_matrix.std(axis=0)
    abs_mean  = np.abs(mean_coef)

    feat_df = pd.DataFrame({
        "dataset":        dataset_name,
        "feature":        all_names,
        "mean_coef":      mean_coef,
        "std_coef":       std_coef,
        "abs_mean_coef":  abs_mean,
    })
    # strip categorical suffix to recover the base feature group name
    feat_df["feature_group"] = feat_df["feature"].str.replace(r"__.*$", "", regex=True)
    feat_df = feat_df.sort_values("abs_mean_coef", ascending=False).reset_index(drop=True)

    # Group-level importance: max |coef| within the group
    group_df = (feat_df.groupby("feature_group")["abs_mean_coef"]
                .max().sort_values(ascending=False)
                .reset_index()
                .rename(columns={"abs_mean_coef": "group_max_abs_coef"}))
    group_df.insert(0, "dataset", dataset_name)
    group_df["group_rank"] = range(1, len(group_df) + 1)

    print(f"\n[{dataset_name}] Top feature groups (by max |coef| across splits):")
    for _, row in group_df.head(15).iterrows():
        print(f"  {int(row['group_rank']):2d}. {row['feature_group']:<55s}  {row['group_max_abs_coef']:.4f}")

    cis = cindex_per_split
    return (
        {"cindex_mean": float(np.mean(cis)), "cindex_std": float(np.std(cis))},
        feat_df,
        group_df,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits",     type=int,   default=5)
    parser.add_argument("--alpha",        type=float, default=0.1,
                        help="Ridge regularisation strength for CoxPH")
    parser.add_argument("--min_cat_freq", type=int,   default=10,
                        help="Min train-set frequency to keep a category (else → OTHER)")
    parser.add_argument("--base",  type=str,
                        default="/data/rbg/users/ashyamal/RNA-FM-benchmarking")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: <base>/benchmarking_results/clinical)")
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(args.base) / "benchmarking_results" / "clinical")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"Full clinical baseline  |  alpha={args.alpha}  |  "
          f"n_splits={args.n_splits}  |  min_cat_freq={args.min_cat_freq}")
    print("=" * 90)

    summary_rows = []
    all_feat_dfs, all_group_dfs = [], []

    for ds in DATASETS:
        print(f"\n{'='*40}\nDataset: {ds}\n{'='*40}")
        result, feat_df, group_df = run_dataset(
            ds, n_splits=args.n_splits, alpha=args.alpha,
            base=args.base, min_cat_freq=args.min_cat_freq,
        )
        summary_rows.append({
            "Dataset":          ds,
            "cindex_mean":      result["cindex_mean"],
            "cindex_std":       result["cindex_std"],
            "cindex_formatted": (
                "NA" if np.isnan(result["cindex_mean"])
                else f"{result['cindex_mean']:.3f} ± {result['cindex_std']:.3f}"
            ),
        })
        if not feat_df.empty:
            all_feat_dfs.append(feat_df)
        if not group_df.empty:
            all_group_dfs.append(group_df)

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("C-INDEX RESULTS")
    print("=" * 90)
    print(summary[["Dataset", "cindex_formatted"]].to_string(index=False))

    summary.to_csv(out_dir / "cindex_results.csv", index=False)
    print(f"\nC-index results  → {out_dir / 'cindex_results.csv'}")

    if all_feat_dfs:
        feat_all = pd.concat(all_feat_dfs, ignore_index=True)
        feat_all.to_csv(out_dir / "feature_importance.csv", index=False)
        print(f"Feature importance  → {out_dir / 'feature_importance.csv'}")

    if all_group_dfs:
        group_all = pd.concat(all_group_dfs, ignore_index=True)
        group_all.to_csv(out_dir / "feature_group_importance.csv", index=False)
        print(f"Group importance    → {out_dir / 'feature_group_importance.csv'}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASETS_DIR = os.environ.get("RAW_DATASETS_DIR", "/data/rbg/shared/datasets")


# ----------------------- Clinical cleaning ----------------------------------
BAD_VALUES = {"unknown", "not reported", "'--", "not applicable", "n/a", "na", "none", "", "nan/null", "--"}

def clean_clinical(df):
    """Replace any bad string values (case-insensitive) with NaN."""
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


# ----------------------------- CLI ------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_splits", type=int, default=5)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--base", type=str, default=str(REPO_ROOT))
parser.add_argument("--out_csv", type=str, default=str(REPO_ROOT / "benchmarking_results" / "clinical_baseline_results.csv"))
args = parser.parse_args()

BASE = args.base

DATASETS = ["TCGA", "Hartwig", "CPTAC", "TARGET_AML"]

# covariate_mode options:
#   "base"        -- demographic + staging (no organ)
#   "base+organ"  -- demographic + staging + tumor location
#   "organ_only"  -- tumor location only
COVARIATE_MODES = ["base", "base+organ", "organ_only"]

COVARIATE_LABELS = {
    "base":        "Demographic & staging",
    "base+organ":  "Demographic & staging + tumor location",
    "organ_only":  "Tumor location only",
}

stage_map = {
    # Stage 0
    'Stage 0': 0, 'Stage 0a': 0, 'Stage 0is': 0, 'Stage IS': 0,
    # Stage I
    'Stage I': 1, 'Stage IA': 1, 'Stage IA1': 1, 'Stage IA2': 1, 'Stage IA3': 1,
    'Stage IB': 1, 'Stage IB1': 1, 'Stage IC': 1,
    # Stage II
    'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2, 'Stage IIC': 2,
    # Stage III
    'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
    'Stage IIIC1': 3, 'Stage IIIC2': 3,
    # 'Stage X' (cannot be assessed) is intentionally unmapped -> treated as missing
    # Stage IV
    'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4, 'Stage IVC': 4,
}


# ----------------------- Organ encoding helpers -----------------------------
def build_organ_map(patient_data, ids, unk_token="__UNK__"):
    organs = sorted({str(patient_data[s].get("organ"))
                     for s in ids if s in patient_data
                     and patient_data[s].get("organ") is not None
                     and not pd.isnull(patient_data[s].get("organ"))})
    organ_to_idx = {org: i for i, org in enumerate(organs)}
    organ_to_idx[unk_token] = len(organ_to_idx)
    return organ_to_idx, unk_token


def organ_onehot(organ_str, organ_to_idx, unk_token="__UNK__"):
    v = np.zeros(len(organ_to_idx), dtype=float)
    if organ_str is None or pd.isnull(organ_str):
        v[organ_to_idx[unk_token]] = 1.0
    else:
        v[organ_to_idx.get(str(organ_str), organ_to_idx[unk_token])] = 1.0
    return v


# ----------------------- Dataset loading ------------------------------------
def load_dataset(dataset_name):
    """
    Returns (patient_data, clinical_dedup, split_dir, sample_to_patient).

    sample_to_patient is used ONLY for deduplication:
      - Hartwig: sample ID -> patient ID (clinical still keyed by sample ID)
      - TARGET_AML: built per-split by stripping last suffix
      - others: None
    """
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
            f"{RAW_DATASETS_DIR}/CPTAC/clinical/clinical.tsv", sep="\t", low_memory=False
        ))
        clinical_dedup = dedup_gdc_clinical(clinical, "cases.case_id")
        # demographic.age_at_index is empty for CPTAC; age_at_diagnosis is in days
        clinical_dedup["age_years"] = (
            pd.to_numeric(clinical_dedup["diagnoses.age_at_diagnosis"], errors="coerce") / 365.25
        )
        return patient_data, clinical_dedup, f"{BASE}/splits/CPTAC", None

    elif dataset_name == "Hartwig":
        with open(f"{BASE}/data/RNA_embeddings/Hartwig_processed.pkl", "rb") as f:
            patient_data = pickle.load(f)
        with open(f"{BASE}/data/Hartwig/sample_to_patient_mapping.pkl", "rb") as f:
            sample_to_patient = pickle.load(f)
        metadata = pd.read_csv(
            f"{RAW_DATASETS_DIR}/Hartwig/isofox_tpm_update_4/metadata.tsv", sep="\t"
        )
        metadata["biopsyDate"] = pd.to_datetime(metadata["biopsyDate"])
        metadata["age_at_biopsy"] = metadata["biopsyDate"].dt.year - metadata["birthYear"]
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

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ----------------------- ID helpers -----------------------------------------
def deduplicate_by_patient(sample_ids, s2p):
    """Keep first sample per patient, preserving order."""
    seen, result = set(), []
    for sid in sample_ids:
        pid = s2p.get(sid, sid) if s2p else sid
        if pid not in seen:
            seen.add(pid)
            result.append(sid)
    return result


def get_clinical_id(sid, dataset_name):
    """Returns the key used to look up sid in clinical_dedup."""
    if dataset_name == "TARGET_AML":
        return "-".join(sid.rsplit("-", 1)[:1])
    return sid


# ----------------------- Feature extraction ---------------------------------
def get_base_features(dataset_name, row):
    """Returns (feats, valid) for demographic + staging features only."""
    if dataset_name == "TCGA":
        age    = row.get("demographic.age_at_index", np.nan)
        gender = row.get("demographic.gender", np.nan)
        stage  = row.get("diagnoses.ajcc_pathologic_stage", np.nan)
        if pd.isnull(age) or pd.isnull(gender) or stage not in stage_map:
            return None, False
        return np.array([float(age), 1.0 if gender == "male" else 0.0, float(stage_map[stage])]), True

    elif dataset_name == "CPTAC":
        age    = row.get("age_years", np.nan)
        gender = row.get("demographic.gender", np.nan)
        stage  = row.get("diagnoses.ajcc_pathologic_stage", np.nan)
        if pd.isnull(age) or pd.isnull(gender) or stage not in stage_map:
            return None, False
        return np.array([float(age), 1.0 if gender == "male" else 0.0, float(stage_map[stage])]), True

    elif dataset_name == "Hartwig":
        age    = row.get("age_at_biopsy", np.nan)
        gender = row.get("gender", np.nan)
        if pd.isnull(age) or pd.isnull(gender):
            return None, False
        return np.array([float(age), 1.0 if gender == "MALE" else 0.0]), True

    elif dataset_name == "TARGET_AML":
        age    = row.get("demographic.age_at_index", np.nan)
        gender = row.get("demographic.gender", np.nan)
        if pd.isnull(age) or pd.isnull(gender):
            return None, False
        return np.array([float(age), 1.0 if str(gender).lower() == "male" else 0.0]), True


# ----------------------- Common valid ID filtering --------------------------
def get_valid_ids(dataset_name, patient_data, clinical_dedup, ids):
    """
    Returns the subset of ids that are valid for ALL covariate modes:
      - present in patient_data
      - present in clinical_dedup
      - base features valid (age, gender, stage where applicable)
      - organ not missing (so organ_only mode uses same patients)
      - PFS time and event present and non-negative
    For TARGET_AML, organ check is skipped (all same site, organ_only not run).
    """
    valid = []
    for sid in ids:
        if sid not in patient_data:
            continue
        cid = get_clinical_id(sid, dataset_name)
        if cid not in clinical_dedup.index:
            continue

        # Check base features
        _, base_valid = get_base_features(dataset_name, clinical_dedup.loc[cid])
        if not base_valid:
            continue

        # Check organ (skip for TARGET_AML since organ_only not run there)
        if dataset_name != "TARGET_AML":
            organ = patient_data[sid].get("organ", None)
            if organ is None or pd.isnull(organ):
                continue

        # Check labels
        pfs_time  = patient_data[sid].get("PFS_time", None)
        pfs_event = patient_data[sid].get("PFS_event", None)
        if pfs_time is None or pfs_event is None or float(pfs_time) < 0:
            continue

        valid.append(sid)
    return valid


# ----------------------- Summary stats --------------------------------------
def print_dataset_summary(dataset_name, patient_data, clinical_dedup, all_sample_ids):
    ages, genders, stages = [], [], []
    n_missing_data = n_missing_clinical = n_kept = 0

    for sid in all_sample_ids:
        if sid not in patient_data:
            n_missing_data += 1
            continue
        cid = get_clinical_id(sid, dataset_name)
        if cid not in clinical_dedup.index:
            n_missing_clinical += 1
            continue
        feats, valid = get_base_features(dataset_name, clinical_dedup.loc[cid])
        if not valid:
            continue
        n_kept += 1

        row = clinical_dedup.loc[cid]
        if dataset_name == "TCGA":
            ages.append(float(row.get("demographic.age_at_index")))
            genders.append(str(row.get("demographic.gender")))
            stages.append(str(row.get("diagnoses.ajcc_pathologic_stage")))
        elif dataset_name == "CPTAC":
            ages.append(float(row.get("age_years")))
            genders.append(str(row.get("demographic.gender")))
            stages.append(str(row.get("diagnoses.ajcc_pathologic_stage")))
        elif dataset_name == "Hartwig":
            ages.append(float(row.get("age_at_biopsy")))
            genders.append(str(row.get("gender")))
        elif dataset_name == "TARGET_AML":
            ages.append(float(row.get("demographic.age_at_index")))
            genders.append(str(row.get("demographic.gender")))

    print(f"\n{'='*60}")
    print(f"  Dataset summary: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total unique sample IDs (across all splits): {len(all_sample_ids)}")
    print(f"  Missing from patient_data:                   {n_missing_data}")
    print(f"  Missing from clinical:                       {n_missing_clinical}")
    print(f"  Patients kept after filtering:               {n_kept}")
    if ages:
        print(f"\n  Age:  {len(set(ages))} unique values, "
              f"range {min(ages):.0f}–{max(ages):.0f}, "
              f"mean {np.mean(ages):.1f} ± {np.std(ages):.1f}")
    if genders:
        print(f"\n  Gender:")
        for val, count in pd.Series(genders).value_counts().items():
            print(f"    {val:<20} n={count}")
    if stages:
        print(f"\n  Tumor stage:")
        for val, count in pd.Series(stages).value_counts().items():
            print(f"    {val:<20} n={count}")
    print(f"{'='*60}\n")


# ----------------------- prepare_Xy -----------------------------------------
def prepare_Xy(dataset_name, patient_data, clinical_dedup, ids, covariate_mode,
               organ_to_idx=None, unk_token="__UNK__"):
    """
    ids should already be pre-filtered by get_valid_ids, so no patient-level
    filtering is needed here beyond assembling features.
    """
    X, y = [], []
    for sid in ids:
        organ = patient_data[sid].get("organ", None)
        cid   = get_clinical_id(sid, dataset_name)
        row   = clinical_dedup.loc[cid]

        if covariate_mode == "organ_only":
            feats = organ_onehot(organ, organ_to_idx, unk_token)
        else:
            feats, _ = get_base_features(dataset_name, row)
            if covariate_mode == "base+organ":
                feats = np.concatenate([feats, organ_onehot(organ, organ_to_idx, unk_token)])

        X.append(feats)
        y.append((bool(patient_data[sid]["PFS_event"]), float(patient_data[sid]["PFS_time"])))

    if len(X) == 0:
        return np.zeros((0, 0)), np.array([], dtype=[("event", bool), ("time", float)])
    return np.vstack(X), np.array(y, dtype=[("event", bool), ("time", float)])


# ----------------------------- Runner ---------------------------------------
def run_dataset(dataset_name, covariate_mode, n_splits, alpha,
                precomputed_valid=None):
    """
    precomputed_valid: dict mapping split index -> (train_ids, test_ids)
    with the common valid patient sets. Computed on first call and reused.
    """
    # organ_only not meaningful for TARGET_AML
    if covariate_mode == "organ_only" and dataset_name == "TARGET_AML":
        return {"cindex_mean": np.nan, "cindex_std": np.nan, "split_details": []}, precomputed_valid

    patient_data, clinical_dedup, split_dir, sample_to_patient = load_dataset(dataset_name)

    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    all_sample_ids = set()
    split_details  = []
    needs_organ    = covariate_mode in ("base+organ", "organ_only")

    # Compute valid IDs once (on first mode) and reuse for subsequent modes
    if precomputed_valid is None:
        precomputed_valid = {}

    for i in range(n_splits):
        split_path = split_dir / f"split_{i}.pkl"
        if not split_path.exists():
            print(f"[{dataset_name}] WARNING: missing {split_path}, skipping")
            continue

        if i not in precomputed_valid:
            with open(split_path, "rb") as f:
                split = pickle.load(f)

            if dataset_name == "TARGET_AML":
                s2p = {s: "-".join(s.rsplit("-", 1)[:1]) for s in split["train"] + split["test"]}
            elif dataset_name == "Hartwig":
                s2p = sample_to_patient
            else:
                s2p = None

            raw_train = deduplicate_by_patient(split["train"], s2p)
            raw_test  = deduplicate_by_patient(split["test"],  s2p)

            def patient_ids(ids): return [s2p.get(s, s) if s2p else s for s in ids]
            assert len(set(patient_ids(raw_train))) == len(raw_train), \
                f"[{dataset_name}] Split {i}: duplicate patient IDs in train"
            assert len(set(patient_ids(raw_test))) == len(raw_test), \
                f"[{dataset_name}] Split {i}: duplicate patient IDs in test"

            # Filter to patients valid for ALL modes
            train_ids = get_valid_ids(dataset_name, patient_data, clinical_dedup, raw_train)
            test_ids  = get_valid_ids(dataset_name, patient_data, clinical_dedup, raw_test)
            precomputed_valid[i] = (train_ids, test_ids)
        else:
            train_ids, test_ids = precomputed_valid[i]

        all_sample_ids.update(train_ids)
        all_sample_ids.update(test_ids)

        organ_to_idx, unk_token = (build_organ_map(patient_data, train_ids) if needs_organ
                                   else (None, None))

        X_train, y_train = prepare_Xy(dataset_name, patient_data, clinical_dedup,
                                      train_ids, covariate_mode, organ_to_idx, unk_token)
        X_test,  y_test  = prepare_Xy(dataset_name, patient_data, clinical_dedup,
                                      test_ids,  covariate_mode, organ_to_idx, unk_token)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"[{dataset_name}|{covariate_mode}] Split {i}: empty after filtering, skipping")
            continue

        model = CoxPHSurvivalAnalysis(alpha=alpha)
        model.fit(X_train, y_train)
        ci = float(concordance_index_censored(y_test["event"], y_test["time"],
                                              model.predict(X_test))[0])
        split_details.append((ci, len(X_train), len(X_test)))
        print(f"[{dataset_name}|{covariate_mode}] Split {i}: C-index={ci:.4f} "
              f"(n_train={len(X_train)}, n_test={len(X_test)})")

    if covariate_mode == "base":
        print_dataset_summary(dataset_name, patient_data, clinical_dedup, list(all_sample_ids))

    if not split_details:
        return {"cindex_mean": np.nan, "cindex_std": np.nan, "split_details": []}, precomputed_valid

    cis = [x[0] for x in split_details]
    return {
        "cindex_mean":   float(np.mean(cis)),
        "cindex_std":    float(np.std(cis)),
        "split_details": split_details,
    }, precomputed_valid


# ----------------------- Feature value counts -------------------------------
def dump_feature_value_counts(dataset_name, patient_data, clinical_dedup, out_path):
    if dataset_name == "TCGA":
        fields = ["demographic.age_at_index", "demographic.gender", "diagnoses.ajcc_pathologic_stage"]
    elif dataset_name == "CPTAC":
        fields = ["age_years", "demographic.gender", "diagnoses.ajcc_pathologic_stage"]
    elif dataset_name == "Hartwig":
        fields = ["age_at_biopsy", "gender"]
    elif dataset_name == "TARGET_AML":
        fields = ["demographic.age_at_index", "demographic.gender"]
    else:
        return

    counts = {f: {} for f in fields}
    counts["organ"] = {}

    for sid in patient_data:
        cid = get_clinical_id(sid, dataset_name)
        if cid not in clinical_dedup.index:
            continue
        row = clinical_dedup.loc[cid]
        for f in fields:
            val = str(row.get(f, "MISSING")) if not pd.isnull(row.get(f, np.nan)) else "NaN/null"
            counts[f][val] = counts[f].get(val, 0) + 1
        organ = patient_data[sid].get("organ", None)
        organ_str = "NaN/null" if organ is None or pd.isnull(organ) else str(organ)
        counts["organ"][organ_str] = counts["organ"].get(organ_str, 0) + 1

    # sort by count descending
    counts = {f: dict(sorted(v.items(), key=lambda x: -x[1])) for f, v in counts.items()}

    with open(out_path, "w") as f:
        json.dump({dataset_name: counts}, f, indent=2)
    print(f"[{dataset_name}] Feature value counts saved to {out_path}")


# ----------------------------- Main -----------------------------------------
def main():
    print("=" * 90)
    print(f"Running CoxPH clinical baselines | datasets={DATASETS} | "
          f"alpha={args.alpha} | n_splits={args.n_splits}")
    print("=" * 90)

    rows = []
    for ds in DATASETS:
        print(f"\n{'='*40}\nDataset: {ds}\n{'='*40}")
        _pd, _cd, _, _ = load_dataset(ds)
        dump_feature_value_counts(
            ds, _pd, _cd,
            f"{BASE}/benchmarking_results/feature_counts/feature_counts_{ds}.json"
        )
        precomputed_valid = None  # shared across modes for this dataset
        for mode in COVARIATE_MODES:
            out, precomputed_valid = run_dataset(
                ds, covariate_mode=mode, n_splits=args.n_splits,
                alpha=args.alpha, precomputed_valid=precomputed_valid
            )
            mean, std   = out["cindex_mean"], out["cindex_std"]
            details     = out["split_details"]

            rows.append({
                "Dataset":           ds,
                "Covariates":        COVARIATE_LABELS[mode],
                "covariate_mode":    mode,
                "cindex_mean":       mean,
                "cindex_std":        std,
                "cindex_formatted":  "NA" if np.isnan(mean) else f"{mean:.3f} ± {std:.3f}",
                "per_split_cindex":  ", ".join(f"{ci:.4f}" for ci, _, _ in details),
                "per_split_n_train": ", ".join(str(n_tr) for _, n_tr, _ in details),
                "per_split_n_test":  ", ".join(str(n_te) for _, _, n_te in details),
                "per_split_detail":  ", ".join(
                    f"{ci:.4f} (train={n_tr} test={n_te})"
                    for ci, n_tr, n_te in details
                ),
            })

    df = pd.DataFrame(rows)

    wide = df.pivot_table(index="Covariates", columns="Dataset",
                          values="cindex_formatted", aggfunc="first")
    wide = wide.reindex(columns=DATASETS)
    wide = wide.reindex([COVARIATE_LABELS[m] for m in COVARIATE_MODES])

    print("\n" + "=" * 90)
    print("RESULTS (formatted for table)")
    print("=" * 90)
    print(wide.to_string())

    df.to_csv(args.out_csv, index=False)
    print(f"\nFull results saved to: {args.out_csv}")

    wide_path = args.out_csv.replace(".csv", "_wide.csv")
    wide.to_csv(wide_path)
    print(f"Wide format saved to:  {wide_path}")


if __name__ == "__main__":
    main()
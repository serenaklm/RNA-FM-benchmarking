import os 
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import argparse
import numpy as np
from collections import Counter

SEED_IDX = {0: 17,
            1: 37,
            2: 68,
            3: 168,
            4: 99}

def load_pickle(path):
    with open(path, "rb") as f: 
        return pickle.load(f)

def pickle_data(data, path):
    with open(path, "wb") as f: 
        pickle.dump(data, f)

def split_TCGA(
    data_path,
    split_folder,
    n_splits=5,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_key="cancer_type",  # <- uses data[pid][stratify_key]
):
    data = load_pickle(data_path)
    all_patient_ids = list(data.keys())

    n_train = int(len(all_patient_ids) * train_ratio)
    n_val = int(len(all_patient_ids) * val_ratio)
    n_test = len(all_patient_ids) - n_train - n_val
    assert n_test > 0 and n_val > 0 and n_train > 0

    # Build stratification labels (e.g., cancer_type)
    labels = []
    missing = []
    for pid in all_patient_ids:
        lab = data[pid].get(stratify_key, None)
        if lab is None:
            missing.append(pid)
        labels.append(lab)

    if missing:
        raise KeyError(
            f"Missing '{stratify_key}' for {len(missing)} patients (e.g., {missing[:5]}). "
            "Needed for stratified splits."
        )

    for idx in range(n_splits):
        output_path = os.path.join(split_folder, f"split_{idx}.pkl")
        if os.path.exists(output_path):
            print(f"{output_path} already exists. Not creating a new split")
            continue

        seed = SEED_IDX[idx]

        # Train vs temp (val+test), stratified by label
        train_ids, temp_ids, y_train, y_temp = train_test_split(
            all_patient_ids,
            labels,
            train_size=n_train,
            test_size=(n_val + n_test),
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )

        # Val vs test from temp, stratified by label
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=n_val,
            test_size=n_test,
            random_state=seed,
            shuffle=True,
            stratify=y_temp,
        )

        assert len(train_ids) == n_train, (len(train_ids), n_train)
        assert len(val_ids) == n_val, (len(val_ids), n_val)
        assert len(test_ids) == n_test, (len(test_ids), n_test)

        assert len(set(train_ids).intersection(val_ids)) == 0
        assert len(set(train_ids).intersection(test_ids)) == 0
        assert len(set(val_ids).intersection(test_ids)) == 0

        split = {"train": train_ids, "val": val_ids, "test": test_ids}
        print(f"idx:{idx}", len(train_ids), len(val_ids), len(test_ids))
        pickle_data(split, output_path)

def split_CPTAC(
    data_path,
    split_folder,
    n_splits=5,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_key="organ",
    min_samples_per_class=10,  # Minimum samples needed to stratify on a class
):
    """
    CPTAC split stratified by organ (similar to TCGA stratified by cancer_type).
    Groups rare organs (<min_samples_per_class) into "Other" category.
    """
    data = load_pickle(data_path)
    all_ids = list(data.keys())

    n_train = int(len(all_ids) * train_ratio)
    n_val = int(len(all_ids) * val_ratio)
    n_test = len(all_ids) - n_train - n_val
    assert n_test > 0 and n_val > 0 and n_train > 0

    # Build stratification labels (organ)
    labels = []
    missing = []
    for pid in all_ids:
        lab = data[pid].get(stratify_key, None)
        if lab is None:
            missing.append(pid)
            lab = "Unknown"
        labels.append(lab)

    if missing:
        print(f"Warning: Missing '{stratify_key}' for {len(missing)} patients. Using 'Unknown'.")

    # Count samples per organ
    from collections import Counter
    organ_counts = Counter(labels)
    print(f"Organ distribution: {dict(organ_counts)}")

    # Group rare organs into "Other"
    rare_organs = {organ for organ, count in organ_counts.items() if count < min_samples_per_class}
    if rare_organs:
        print(f"Grouping rare organs (< {min_samples_per_class} samples) into 'Other': {rare_organs}")
        labels = [lab if lab not in rare_organs else "Other" for lab in labels]
        organ_counts = Counter(labels)
        print(f"Organ distribution after grouping: {dict(organ_counts)}")

    # Check if stratification is still possible
    can_stratify = all(count >= 2 for count in organ_counts.values())
    if not can_stratify:
        print("Warning: Cannot stratify (some classes still have < 2 samples). Using random split.")
        labels = None

    for idx in range(n_splits):
        output_path = os.path.join(split_folder, f"split_{idx}.pkl")
        if os.path.exists(output_path):
            print(f"{output_path} already exists. Not creating a new split")
            continue

        seed = SEED_IDX[idx]

        if labels is None:
            # Unstratified split
            train_ids, temp_ids = train_test_split(
                all_ids,
                train_size=n_train,
                test_size=(n_val + n_test),
                random_state=seed,
                shuffle=True,
            )
            val_ids, test_ids = train_test_split(
                temp_ids,
                train_size=n_val,
                test_size=n_test,
                random_state=seed,
                shuffle=True,
            )
        else:
            # Stratified split
            train_ids, temp_ids, y_train, y_temp = train_test_split(
                all_ids,
                labels,
                train_size=n_train,
                test_size=(n_val + n_test),
                random_state=seed,
                shuffle=True,
                stratify=labels,
            )
            val_ids, test_ids = train_test_split(
                temp_ids,
                train_size=n_val,
                test_size=n_test,
                random_state=seed,
                shuffle=True,
                stratify=y_temp,
            )

        assert len(train_ids) == n_train, (len(train_ids), n_train)
        assert len(val_ids) == n_val, (len(val_ids), n_val)
        assert len(test_ids) == n_test, (len(test_ids), n_test)

        assert len(set(train_ids).intersection(val_ids)) == 0
        assert len(set(train_ids).intersection(test_ids)) == 0
        assert len(set(val_ids).intersection(test_ids)) == 0

        split = {"train": train_ids, "val": val_ids, "test": test_ids}
        print(f"idx:{idx}", len(train_ids), len(val_ids), len(test_ids))
        pickle_data(split, output_path)

def split_TARGET_AML(
    data_path,
    split_folder,
    n_splits=5,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
):
    data = load_pickle(data_path)
    all_sample_ids = list(data.keys())

    # Group samples by patient_id
    from collections import defaultdict
    patient_to_samples = defaultdict(list)
    missing_patient = []

    for sid in all_sample_ids:
        pid = data[sid].get("patient_id", None)
        if pid is None:
            missing_patient.append(sid)
            continue
        patient_to_samples[pid].append(sid)

    if missing_patient:
        raise KeyError(
            f"Missing 'patient_id' for {len(missing_patient)} samples "
            f"(e.g. {missing_patient[:5]})."
        )

    all_patient_ids = list(patient_to_samples.keys())

    n_train = int(len(all_patient_ids) * train_ratio)
    n_val = int(len(all_patient_ids) * val_ratio)
    n_test = len(all_patient_ids) - n_train - n_val
    assert n_test > 0 and n_val > 0 and n_train > 0

    print(f"[TARGET_AML] Total samples: {len(all_sample_ids)}")
    print(f"[TARGET_AML] Unique patients: {len(all_patient_ids)}")

    for idx in range(n_splits):
        output_path = os.path.join(split_folder, f"split_{idx}.pkl")
        if os.path.exists(output_path):
            print(f"{output_path} already exists. Not creating a new split")
            continue

        seed = SEED_IDX[idx]

        # Random split on patients
        train_pids, temp_pids = train_test_split(
            all_patient_ids,
            train_size=n_train,
            test_size=(n_val + n_test),
            random_state=seed,
            shuffle=True,
        )
        val_pids, test_pids = train_test_split(
            temp_pids,
            train_size=n_val,
            test_size=n_test,
            random_state=seed,
            shuffle=True,
        )

        # Expand patients -> samples
        train_ids = [sid for pid in train_pids for sid in patient_to_samples[pid]]
        val_ids   = [sid for pid in val_pids   for sid in patient_to_samples[pid]]
        test_ids  = [sid for pid in test_pids  for sid in patient_to_samples[pid]]

        # Sample-level sanity checks
        assert len(set(train_ids).intersection(val_ids)) == 0
        assert len(set(train_ids).intersection(test_ids)) == 0
        assert len(set(val_ids).intersection(test_ids)) == 0

        # Patient-level sanity checks
        assert len(set(train_pids).intersection(val_pids)) == 0
        assert len(set(train_pids).intersection(test_pids)) == 0
        assert len(set(val_pids).intersection(test_pids)) == 0

        split = {"train": train_ids, "val": val_ids, "test": test_ids}
        print(
            f"[TARGET_AML] idx:{idx} "
            f"patients: {len(train_pids)}/{len(val_pids)}/{len(test_pids)} | "
            f"samples: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
        )
        pickle_data(split, output_path)

def split_Hartwig(
    data_path,
    split_root_folder,
    n_splits=5,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_key="cancer_type",
    sample_to_patient_path="/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/Hartwig/sample_to_patient_dict.pkl",
    forbidden_patient_ids=None,
):
    data = load_pickle(data_path)
    sample_to_patient = load_pickle(sample_to_patient_path)

    forbidden = set(forbidden_patient_ids) if forbidden_patient_ids else set()

    all_sample_ids = list(data.keys())

    # Group samples by patient, excluding forbidden patients
    from collections import defaultdict
    patient_to_samples = defaultdict(list)
    for sid in all_sample_ids:
        pid = sample_to_patient[sid]
        if pid in forbidden:
            continue
        patient_to_samples[pid].append(sid)

    all_patient_ids = list(patient_to_samples.keys())

    n_excluded = len(all_sample_ids) - sum(len(v) for v in patient_to_samples.values())
    print(f"[Hartwig] Total samples: {len(all_sample_ids)}, excluded (forbidden): {n_excluded}, remaining: {len(all_sample_ids) - n_excluded}, Unique patients: {len(all_patient_ids)}")

    data_key = Path(data_path).name.replace("_processed.pkl", "")

    def _make_splits(eligible_patient_ids, out_folder: Path, tag: str):
        out_folder.mkdir(exist_ok=True, parents=True)

        n_train = int(len(eligible_patient_ids) * train_ratio)
        n_val   = int(len(eligible_patient_ids) * val_ratio)
        n_test  = len(eligible_patient_ids) - n_train - n_val
        assert n_test > 0 and n_val > 0 and n_train > 0

        # Build stratification labels at patient level
        # Use the first sample's metadata per patient
        labels = []
        missing = []
        for pid in eligible_patient_ids:
            sid = patient_to_samples[pid][0]
            c_type = data[sid].get(stratify_key, None)
            resp   = data[sid].get("response", None)
            if c_type is None or resp is None:
                missing.append(pid)
            labels.append(f"{c_type}_{resp}")

        if missing:
            raise KeyError(f"[{tag}] Missing metadata for {len(missing)} patients.")

        counts = Counter(labels)
        rare = {k for k, v in counts.items() if v < 10}
        if rare:
            labels = [lab if lab not in rare else "Other" for lab in labels]
            if any(v < 10 for v in Counter(labels).values()):
                labels = None

        for idx in range(n_splits):
            output_path = os.path.join(out_folder, f"split_{idx}.pkl")
            if os.path.exists(output_path):
                print(f"{output_path} already exists. Not creating a new split")
                continue

            seed = SEED_IDX[idx]

            if labels is None:
                train_pids, temp_pids = train_test_split(
                    eligible_patient_ids, train_size=n_train, test_size=(n_val + n_test),
                    random_state=seed, shuffle=True,
                )
                val_pids, test_pids = train_test_split(
                    temp_pids, train_size=n_val, test_size=n_test,
                    random_state=seed, shuffle=True,
                )
            else:
                train_pids, temp_pids, _, y_temp = train_test_split(
                    eligible_patient_ids, labels, train_size=n_train, test_size=(n_val + n_test),
                    random_state=seed, shuffle=True, stratify=labels,
                )
                val_pids, test_pids = train_test_split(
                    temp_pids, train_size=n_val, test_size=n_test,
                    random_state=seed, shuffle=True, stratify=y_temp,
                )

            # Expand patients -> samples
            train_ids = [sid for pid in train_pids for sid in patient_to_samples[pid]]
            val_ids   = [sid for pid in val_pids   for sid in patient_to_samples[pid]]
            test_ids  = [sid for pid in test_pids  for sid in patient_to_samples[pid]]

            # Sanity checks (sample level)
            assert len(set(train_ids) & set(val_ids))  == 0
            assert len(set(train_ids) & set(test_ids)) == 0
            assert len(set(val_ids)   & set(test_ids)) == 0

            split = {"train": train_ids, "val": val_ids, "test": test_ids}
            print(f"[{tag}] idx:{idx} patients: {len(train_pids)}/{len(val_pids)}/{len(test_pids)} | samples: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
            pickle_data(split, output_path)

    save_folder = Path(split_root_folder) / f"{data_key}"
    _make_splits(all_patient_ids, save_folder, tag=f"{data_key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"],
    )
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    splits_output_folder = Path("../splits")
    splits_output_folder.mkdir(exist_ok=True, parents=True)

    suffix = ""
    if args.data_set == "TCGA":
        suffix = "_cleaned"
    
    data_path = f"../data/RNA_embeddings/{args.data_set}_processed{suffix}.pkl"
    
    print(f"Loading data from: {data_path}")
    
    if args.data_set not in ["Hartwig", "Hartwig_500"]:
        splits_folder = splits_output_folder / "TCGA_cleaned" if args.data_set == "TCGA" else splits_output_folder / args.data_set
        splits_folder.mkdir(exist_ok=True, parents=True)
        print(f"Saving splits to: {splits_folder}")

    if args.data_set == "TCGA":
        split_TCGA(data_path, splits_folder, n_splits=args.n_splits)
    elif args.data_set == "CPTAC":
        split_CPTAC(data_path, splits_folder, n_splits=args.n_splits)
    elif args.data_set in ["Hartwig", "Hartwig_500"]:
        HARTWIG_FORBIDDEN_PATIENTS = [
            "D00000678", "D00000679", "D00002341", "D00002350", "D00002351",
            "D00002352", "D00002353", "D00002354", "D00001925", "D00004239",
            "D00006712", "D00006922", "D00007741", "D00007765",
        ]
        split_Hartwig(
            data_path=data_path,
            split_root_folder=splits_output_folder,
            n_splits=args.n_splits,
            stratify_key="cancer_type",
            forbidden_patient_ids=HARTWIG_FORBIDDEN_PATIENTS,
        )
    elif args.data_set == "TARGET_AML":
        split_TARGET_AML(
            data_path, 
            splits_folder, 
            n_splits=args.n_splits
        )

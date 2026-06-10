"""
get_ICI_data_splits.py

Generate train/val/test split pickle files for all three ICI evaluation paradigms:

  1. within_cohort  — stratified within a single cohort (70/15/15), 5 splits
  2. loco           — train on N-1 cohorts, test on held-out cohort, 5 splits
                      (splits differ in which 15% of the training pool is used as val)
  3. transfer       — train on cohort A, test on cohort B, 5 splits
                      (splits differ in which 15% of cohort A is used as val)

All three paradigms reference ICI_all_processed.pkl so the DataModule is identical
for every experiment. The split file encodes what goes in train/val/test.

Usage (from repo root):
    python data_preprocessing/get_ICI_data_splits.py
    python data_preprocessing/get_ICI_data_splits.py --paradigm within_cohort
    python data_preprocessing/get_ICI_data_splits.py --paradigm loco
    python data_preprocessing/get_ICI_data_splits.py --paradigm transfer

Output tree:
    splits/ICI/within_cohort/{Cohort}/split_0.pkl  ...  split_4.pkl
    splits/ICI/loco/held_out_{Cohort}/split_0.pkl  ...  split_4.pkl
    splits/ICI/transfer/{Train}_to_{Test}/split_0.pkl ... split_4.pkl
"""

import argparse
import os
from collections import Counter
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
CONFIG_PATH  = REPO_ROOT / "configs" / "immunotherapy_cohorts.yaml"
ICI_PKL_DIR  = REPO_ROOT / "data" / "RNA_embeddings" / "ICI"
SPLITS_ROOT  = REPO_ROOT / "splits" / "ICI"

# 5 different seeds for reproducible but varied splits
SEED_IDX = {0: 17, 1: 37, 2: 68, 3: 168, 4: 99}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _stratify_labels(ids, data, label_key="response", min_per_class=3):
    """
    Build a label list for stratification.
    Falls back to None (unstratified) if any class has < min_per_class samples.
    """
    labels = [data[sid].get(label_key) for sid in ids]
    counts = Counter(labels)
    if any(c < min_per_class for c in counts.values()):
        print(f"    Warning: class counts {dict(counts)} < {min_per_class}, using unstratified split")
        return None
    return labels


def _split_train_val(pool_ids, data, val_ratio=0.15, seed=37):
    """
    Split pool_ids into (train, val) with optional stratification on response label.
    """
    labels = _stratify_labels(pool_ids, data)
    n_val = max(1, int(len(pool_ids) * val_ratio))
    n_train = len(pool_ids) - n_val

    kwargs = dict(
        train_size=n_train,
        test_size=n_val,
        random_state=seed,
        shuffle=True,
    )
    if labels is not None:
        kwargs["stratify"] = labels

    train_ids, val_ids = train_test_split(pool_ids, **kwargs)
    return train_ids, val_ids


def _print_split_summary(tag, train_ids, val_ids, test_ids, data):
    def resp_str(ids):
        rs = [data[i].get("response") for i in ids]
        return f"R={rs.count(1)} NR={rs.count(0)}"
    print(f"    {tag}  train={len(train_ids)}({resp_str(train_ids)})  "
          f"val={len(val_ids)}({resp_str(val_ids)})  "
          f"test={len(test_ids)}({resp_str(test_ids)})")


# ── Paradigm 1: within-cohort ─────────────────────────────────────────────────

def make_within_cohort_splits(data: dict, cohort_names: list, n_splits=5):
    """
    For each cohort: 5 stratified (by response) 70/15/15 splits.
    Uses only that cohort's samples.
    """
    print("\n=== within_cohort splits ===")
    for cohort in cohort_names:
        cohort_ids = [sid for sid, rec in data.items() if rec.get("cohort") == cohort]
        print(f"\n  Cohort: {cohort}  ({len(cohort_ids)} samples)")

        out_dir = SPLITS_ROOT / "within_cohort" / cohort
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = _stratify_labels(cohort_ids, data, min_per_class=2)

        n_test = max(1, int(len(cohort_ids) * 0.15))
        n_val  = max(1, int(len(cohort_ids) * 0.15))
        n_train = len(cohort_ids) - n_val - n_test
        if n_train <= 0:
            print(f"    WARNING: cohort too small for 70/15/15 split, skipping.")
            continue

        for idx in range(n_splits):
            out_path = out_dir / f"split_{idx}.pkl"
            if out_path.exists():
                print(f"    split_{idx} already exists, skipping.")
                continue

            seed = SEED_IDX[idx]
            kwargs_outer = dict(
                train_size=(n_train + n_val),
                test_size=n_test,
                random_state=seed,
                shuffle=True,
            )
            kwargs_inner = dict(
                train_size=n_train,
                test_size=n_val,
                random_state=seed,
                shuffle=True,
            )
            if labels is not None:
                kwargs_outer["stratify"] = labels
                trainval_ids, test_ids, trainval_labels, _ = train_test_split(
                    cohort_ids, labels, **kwargs_outer
                )
                inner_labels = _stratify_labels(trainval_ids, data, min_per_class=2)
                if inner_labels is not None:
                    kwargs_inner["stratify"] = inner_labels
                train_ids, val_ids = train_test_split(trainval_ids, **kwargs_inner)
            else:
                trainval_ids, test_ids = train_test_split(cohort_ids, **kwargs_outer)
                train_ids, val_ids = train_test_split(trainval_ids, **kwargs_inner)

            split = {"train": train_ids, "val": val_ids, "test": test_ids}
            save_pickle(split, out_path)
            _print_split_summary(f"split_{idx}", train_ids, val_ids, test_ids, data)


# ── Paradigm 2: LOCO ──────────────────────────────────────────────────────────

def make_loco_splits(data: dict, cohort_names: list, n_splits=5, val_ratio=0.15):
    """
    For each held-out cohort:
      test  = ALL samples from held-out cohort (fixed across splits)
      train = 85% of the remaining cohorts' samples (stratified on response)
      val   = 15% of the remaining cohorts' samples
    The 5 splits differ in which 15% becomes val (different random seeds).
    """
    print("\n=== loco splits ===")
    for held_out in cohort_names:
        test_ids = [sid for sid, rec in data.items() if rec.get("cohort") == held_out]
        pool_ids = [sid for sid, rec in data.items() if rec.get("cohort") != held_out]
        print(f"\n  Held-out: {held_out}  test={len(test_ids)}  pool={len(pool_ids)}")

        out_dir = SPLITS_ROOT / "loco" / f"held_out_{held_out}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_splits):
            out_path = out_dir / f"split_{idx}.pkl"
            if out_path.exists():
                print(f"    split_{idx} already exists, skipping.")
                continue

            seed = SEED_IDX[idx]
            train_ids, val_ids = _split_train_val(pool_ids, data, val_ratio=val_ratio, seed=seed)

            split = {"train": train_ids, "val": val_ids, "test": test_ids}
            save_pickle(split, out_path)
            _print_split_summary(f"split_{idx}", train_ids, val_ids, test_ids, data)


# ── Paradigm 3: cohort-to-cohort transfer ────────────────────────────────────

def make_transfer_splits(data: dict, cohort_names: list, n_splits=5, val_ratio=0.15):
    """
    For each ordered pair (train_cohort, test_cohort):
      train = 85% of train_cohort samples (stratified on response)
      val   = 15% of train_cohort samples
      test  = ALL samples from test_cohort (fixed across splits)
    The 5 splits differ in which 15% of train_cohort becomes val.
    """
    print("\n=== transfer splits ===")
    for train_cohort in cohort_names:
        for test_cohort in cohort_names:
            if train_cohort == test_cohort:
                continue

            train_pool = [sid for sid, rec in data.items() if rec.get("cohort") == train_cohort]
            test_ids   = [sid for sid, rec in data.items() if rec.get("cohort") == test_cohort]
            print(f"\n  {train_cohort} → {test_cohort}  pool={len(train_pool)}  test={len(test_ids)}")

            out_dir = SPLITS_ROOT / "transfer" / f"{train_cohort}_to_{test_cohort}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(n_splits):
                out_path = out_dir / f"split_{idx}.pkl"
                if out_path.exists():
                    print(f"    split_{idx} already exists, skipping.")
                    continue

                seed = SEED_IDX[idx]
                train_ids, val_ids = _split_train_val(
                    train_pool, data, val_ratio=val_ratio, seed=seed
                )

                split = {"train": train_ids, "val": val_ids, "test": test_ids}
                save_pickle(split, out_path)
                _print_split_summary(f"split_{idx}", train_ids, val_ids, test_ids, data)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate ICI cohort split files for all evaluation paradigms."
    )
    parser.add_argument(
        "--paradigm",
        choices=["within_cohort", "loco", "transfer", "all"],
        default="all",
        help="Which paradigm(s) to generate splits for. Default: all.",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cohort_names = list(cfg["cohorts"].keys())
    print(f"Cohorts from config: {cohort_names}")

    merged_pkl = ICI_PKL_DIR / "ICI_all_processed.pkl"
    if not merged_pkl.exists():
        raise FileNotFoundError(
            f"{merged_pkl} not found.\n"
            "Run first:\n"
            "  python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge"
        )

    print(f"Loading merged pkl: {merged_pkl}")
    data = load_pickle(merged_pkl)
    print(f"Total samples: {len(data)}")

    paradigms = (
        ["within_cohort", "loco", "transfer"]
        if args.paradigm == "all"
        else [args.paradigm]
    )

    if "within_cohort" in paradigms:
        make_within_cohort_splits(data, cohort_names, n_splits=args.n_splits)

    if "loco" in paradigms:
        make_loco_splits(data, cohort_names, n_splits=args.n_splits)

    if "transfer" in paradigms:
        make_transfer_splits(data, cohort_names, n_splits=args.n_splits)

    print("\nDone. Split tree:")
    for p in sorted(SPLITS_ROOT.rglob("split_*.pkl")):
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

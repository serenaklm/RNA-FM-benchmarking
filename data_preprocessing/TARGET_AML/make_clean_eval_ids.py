# data_preprocessing/TARGET_AML/make_clean_eval_ids.py
#
# The TARGET-AML splits group samples by "TARGET-20-XXXXXX", but the same person can also appear
# as "TARGET-21-XXXXXX" (induction-failure sub-cohort), so 10 people have samples in more than one
# partition. 16 people also have both a blood (03A) and a bone-marrow (09A) sample.
#
# This writes, per split, the val and test samples to score so that
#   - no test sample belongs to a person who is in train or val,
#   - no val sample belongs to a person who is in train,
#   - each person contributes one sample (bone marrow preferred).
# Pass the output to the scoring scripts with --eval_ids_file. Models are not retrained.
#
# Usage (from repo root):
#   python -m data_preprocessing.TARGET_AML.make_clean_eval_ids

import json
import pickle
from pathlib import Path

DATA_PATH = Path("./data/RNA_embeddings/TARGET_AML_processed.pkl")
SPLIT_DIR = Path("./splits/TARGET_AML")
OUT_PATH = SPLIT_DIR / "clean_eval_ids.json"


def person(sample_id: str) -> str:
    # TARGET-20-PARBTV-09A -> PARBTV (same person across TARGET-20 and TARGET-21)
    return sample_id.split("-")[2]


def one_per_person(sample_ids):
    chosen = {}
    for sid in sorted(sample_ids):
        p = person(sid)
        if p not in chosen or (sid.split("-")[3].startswith("09") and not chosen[p].split("-")[3].startswith("09")):
            chosen[p] = sid
    return sorted(chosen.values())


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    out = {}
    for split_file in sorted(SPLIT_DIR.glob("split_*.pkl")):
        split_idx = split_file.stem.split("_")[1]
        with open(split_file, "rb") as f:
            split = pickle.load(f)
        ids = {part: [s for s in split[part] if s in data] for part in ("train", "val", "test")}
        persons = {part: {person(s) for s in ids[part]} for part in ids}

        val = one_per_person([s for s in ids["val"] if person(s) not in persons["train"]])
        test = one_per_person([s for s in ids["test"]
                               if person(s) not in persons["train"] | persons["val"]])
        out[split_idx] = {"val": val, "test": test}
        print(f"split {split_idx}: val {len(ids['val'])} -> {len(val)}, test {len(ids['test'])} -> {len(test)}")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

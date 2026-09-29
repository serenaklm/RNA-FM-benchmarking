# test/eval_binned_HVG_models.py
#
# Evaluates trained models from the binned-HVG ablation experiment.
# Mirrors eval_all_RNA_models.py but uses BinnedHVGSurvivalDataModule,
# reconstructing the binning parameters from the run folder name.
#
# Run folder name format (written by train_binned_HVG_model.py):
#   {arch}_{loss}_rnaKey_binned_{strategy}_{n_bins}_nHVGs_{n_hvgs}_seed_{seed}_split_{split}_{ts}
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 python -m test.eval_binned_HVG_models --dataset TCGA
#   CUDA_VISIBLE_DEVICES=1 python -m test.eval_binned_HVG_models --dataset Hartwig
#   CUDA_VISIBLE_DEVICES=2 python -m test.eval_binned_HVG_models --dataset CPTAC
#   CUDA_VISIBLE_DEVICES=3 python -m test.eval_binned_HVG_models --dataset TARGET_AML
#
# Optional overrides:
#   python -m test.eval_binned_HVG_models \
#     --dataset TCGA \
#     --results_dir /path/to/binned_HVG_ablation \
#     --out_csv /path/to/out.csv

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

from pytorch_modules.data.binned_data import BinnedHVGSurvivalDataModule
from pytorch_modules.models.survival_models import RNA_SurvivalModel
from pytorch_modules.utils.utils import load_json

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Regex to parse the run folder name produced by train_binned_HVG_model.py.
# The non-greedy rna_key match stops at the first _nHVGs_ token, yielding
# e.g. rna_key="binned_linear_64", n_hvgs="10000".
# ---------------------------------------------------------------------------
RUN_RE = re.compile(
    r"^(?P<arch>[^_]+)_(?P<loss>[^_]+)_rnaKey_(?P<rna_key>.+?)_nHVGs_(?P<n_hvgs>[\w]+)"
    r"_seed_(?P<seed>\d+)_split_(?P<split>\d+)_"
)

# Validates and decomposes rna_key into strategy + n_bins.
BINNED_RE = re.compile(
    r"^binned_(?P<strategy>linear|quantile|global_linear|global_quantile"
    r"|per_sample_linear|per_sample_quantile)_(?P<n_bins>\d+)$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_run_name(folder_name: str):
    """Return parsed dict or None if folder does not match expected pattern."""
    m = RUN_RE.search(folder_name)
    if not m:
        return None

    d = m.groupdict()
    d["n_hvgs"] = None if d["n_hvgs"] == "None" else int(d["n_hvgs"])
    d["seed"]   = int(d["seed"])
    d["split"]  = int(d["split"])

    bm = BINNED_RE.match(d["rna_key"])
    if not bm:
        return None  # not a binned run

    d["bin_strategy"] = bm.group("strategy")
    d["n_bins"]       = int(bm.group("n_bins"))
    return d


def safe_cindex(event_bool, time, risk):
    event_bool = np.asarray(event_bool, dtype=bool).reshape(-1)
    time       = np.asarray(time, dtype=float).reshape(-1)
    risk       = np.asarray(risk, dtype=float).reshape(-1)
    if len(risk) < 2 or np.sum(event_bool) == 0:
        return float("nan")
    try:
        ci, *_ = concordance_index_censored(event_bool, time, risk)
        return float(ci)
    except Exception:
        return float("nan")


def load_done_keys(out_csv: Path):
    if not out_csv.exists():
        return set(), None
    df = pd.read_csv(out_csv)
    needed = {"arch", "loss", "bin_strategy", "n_bins", "n_hvgs", "seed", "split"}
    if not needed.issubset(df.columns):
        return set(), df
    done = set(
        zip(
            df["arch"].astype(str),
            df["loss"].astype(str),
            df["bin_strategy"].astype(str),
            df["n_bins"].astype(int),
            df["n_hvgs"].apply(lambda x: None if pd.isna(x) else int(x)),
            df["seed"].astype(int),
            df["split"].astype(int),
        )
    )
    return done, df


def _consolidate(pred_batches):
    ids, risks, times, events, organs, cts = [], [], [], [], [], []
    for batch in pred_batches:
        id_, risk, time, event, organ, cancer_type = batch
        ids.extend(list(id_))
        risks.extend(list(risk))
        times.extend(list(time))
        events.extend(list(event))
        organs.extend(list(organ))
        cts.extend(list(cancer_type))
    return ids, risks, times, events, organs, cts


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def eval_one_folder(
    model_folder: Path,
    dataset: str,
    meta: dict,
    config: dict,
    use_gpu: bool,
    min_n: int,
    min_events: int,
):
    """Load checkpoint, run predict on test split, return results dict."""
    ckpts = sorted(f for f in os.listdir(model_folder) if f.endswith(".ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {model_folder}")

    pl.seed_everything(meta["seed"], workers=True)

    endpoint   = config["data"].get("endpoint", "PFS")
    batch_size = config["data"].get("batch_size", 128)
    num_workers = config["data"].get("num_workers", 4)

    # Reconstruct data module — bin edges are refit on the training split,
    # exactly as during training (same split_idx, same training patients).
    dm = BinnedHVGSurvivalDataModule(
        data_path    = config["data_paths"][dataset],
        split_folder = config["split_folders"][dataset],
        split_idx    = meta["split"],
        endpoint     = endpoint,
        n_HVGs       = meta["n_hvgs"],
        n_bins       = meta["n_bins"],
        bin_strategy = meta["bin_strategy"],
        batch_size   = batch_size,
        num_workers  = num_workers,
    )
    dm.prepare_data()
    dm.setup()

    model = RNA_SurvivalModel.load_from_checkpoint(
        str(model_folder / ckpts[0])
    )
    model.eval()

    trainer_kwargs = {k: v for k, v in config.get("trainer", {}).items()
                     if k not in ("devices", "accelerator", "strategy")}
    accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
    trainer = pl.Trainer(
        default_root_dir=str(model_folder),
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    with torch.no_grad():
        pred_batches = trainer.predict(model, dataloaders=dm.test_dataloader())

    _, risk, time, event, organ, cancer_type = _consolidate(pred_batches)
    event_bool = np.asarray(event, dtype=bool)
    risk       = np.asarray(risk,  dtype=float)
    time       = np.asarray(time,  dtype=float)

    results = {"overall": safe_cindex(event_bool, time, risk)}

    for ct in set(cancer_type):
        if ct is None:
            continue
        idx = [i for i, c in enumerate(cancer_type) if c == ct]
        if len(idx) < min_n or int(np.sum(event_bool[idx])) < min_events:
            continue
        results[str(ct)] = safe_cindex(event_bool[idx], time[idx], risk[idx])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dataset = args.dataset

    base = (
        REPO_ROOT / "model_outputs"
        / dataset / "survival" / "PFS" / "binned_HVG_ablation"
    )
    results_dir = Path(args.results_dir) if args.results_dir else base
    out_csv     = Path(args.out_csv)     if args.out_csv     else (base / "binned_HVG_metrics.csv")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()],
                     key=lambda p: p.name)
    print(f"Dataset={dataset}  |  {len(folders)} subfolders in {results_dir}")

    done_keys, existing_df = load_done_keys(out_csv)
    if existing_df is not None:
        print(f"Loaded {len(done_keys)} existing entries from {out_csv}")
    else:
        print("No existing CSV found — starting fresh")

    new_rows, skipped = [], 0

    for i, folder in tqdm(enumerate(folders, 1), total=len(folders)):
        meta = parse_run_name(folder.name)
        if meta is None:
            continue

        config_path = folder / "config.json"
        if not config_path.exists():
            if args.verbose:
                print(f"SKIP (no config.json): {folder.name}")
            continue

        done_key = (
            meta["arch"],
            meta["loss"],
            meta["bin_strategy"],
            meta["n_bins"],
            meta["n_hvgs"],
            meta["seed"],
            meta["split"],
        )
        if done_key in done_keys:
            skipped += 1
            if args.verbose:
                print(f"SKIP (already in CSV): {done_key}")
            continue

        config = load_json(config_path)

        try:
            res = eval_one_folder(
                model_folder=folder,
                dataset=dataset,
                meta=meta,
                config=config,
                use_gpu=True,
                min_n=args.min_n,
                min_events=args.min_events,
            )
        except Exception as e:
            res = {"overall": np.nan}
            row = {
                "run_dir":      str(folder),
                "dataset":      dataset,
                "arch":         meta["arch"],
                "loss":         meta["loss"],
                "bin_strategy": meta["bin_strategy"],
                "n_bins":       meta["n_bins"],
                "n_hvgs":       meta["n_hvgs"],
                "seed":         meta["seed"],
                "split":        meta["split"],
                "overall":      np.nan,
                "error":        str(e),
            }
            new_rows.append(row)
            print(f"  ERROR {folder.name}: {e}")
            continue

        row = {
            "run_dir":      str(folder),
            "dataset":      dataset,
            "arch":         meta["arch"],
            "loss":         meta["loss"],
            "bin_strategy": meta["bin_strategy"],
            "n_bins":       meta["n_bins"],
            "n_hvgs":       meta["n_hvgs"],
            "seed":         meta["seed"],
            "split":        meta["split"],
            "overall":      res.get("overall", np.nan),
            "error":        "",
        }
        for k, v in res.items():
            if k != "overall":
                row[k] = v

        new_rows.append(row)
        if args.verbose:
            print(f"[{i}/{len(folders)}] {folder.name}  overall={row['overall']:.4f}")

    print(f"Skipped {skipped} already-evaluated runs")

    df_new = pd.DataFrame(new_rows)
    df_out = (
        df_new if existing_df is None
        else pd.concat([existing_df, df_new], ignore_index=True, sort=False)
    )

    base_cols  = ["dataset", "arch", "loss", "bin_strategy", "n_bins",
                  "n_hvgs", "seed", "split", "overall"]
    extra_cols = ["run_dir", "error"]
    ct_cols    = sorted(c for c in df_out.columns if c not in set(base_cols + extra_cols))
    cols       = base_cols + ct_cols + extra_cols

    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} rows → {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate binned-HVG ablation runs and collect C-indices."
    )
    p.add_argument("--dataset", required=True,
                   choices=["TCGA", "CPTAC", "Hartwig", "TARGET_AML"])
    p.add_argument("--results_dir", default=None,
                   help="Folder containing run subdirectories "
                        "(default: model_outputs/{dataset}/survival/PFS/binned_HVG_ablation)")
    p.add_argument("--out_csv", default=None,
                   help="Output CSV path (default: <results_dir>/binned_HVG_metrics.csv)")
    p.add_argument("--min_n",      type=int, default=0,
                   help="Min test-set samples for per-cancer-type C-index.")
    p.add_argument("--min_events", type=int, default=0,
                   help="Min events for per-cancer-type C-index.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)

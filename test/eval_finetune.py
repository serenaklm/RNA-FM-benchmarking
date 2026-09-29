# test/eval_finetune.py
#
# Evaluate fine-tuned FM survival models, writing per-split C-index metrics
# to a CSV in the same format as eval_all_RNA_models.py.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=5 python -m test.eval_finetune --dataset TCGA  --endpoint PFS
#   CUDA_VISIBLE_DEVICES=1 python -m test.eval_finetune --dataset CPTAC --endpoint PFS
#
# Assumes each run folder contains: config.json + one .ckpt
# predict_step returns: (id_, risk, time, event, organ, cancer_type)

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

from pytorch_modules.utils.utils import load_json
from pytorch_modules.data.finetune_data import (
    FinetuneTokensSurvivalDataModule,
    scgpt_collate_fn,
    cancerfoundation_collate_fn,
)
from pytorch_modules.models.finetune_models import FM_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]


# Matches both old format: finetune_{fm_name}_{loss}_seed_{seed}_split_{split}_{ts}
# and new format:          finetune_{fm_name}_{loss}_{ft_mode}_seed_{seed}_split_{split}_{ts}
# ft_mode is all-letters (FFT, PFT, LFT, LoRA) so it's distinguishable from "seed"
RUN_RE = re.compile(
    r"^finetune_(?P<fm_name>[^_]+)_(?P<loss>[^_]+)_(?:(?P<ft_mode_dir>[A-Za-z]+)_)?seed_(?P<seed>\d+)_split_(?P<split>\d+)_"
)


def safe_cindex(event_bool, time, risk):
    event_bool = np.asarray(event_bool, dtype=bool).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    if len(risk) < 2 or np.sum(event_bool) == 0:
        return float("nan")
    try:
        cindex, _, _, _, _ = concordance_index_censored(event_bool, time, risk)
        return float(cindex)
    except Exception:
        return float("nan")


def parse_run_name(folder_name: str):
    m = RUN_RE.search(folder_name)
    if not m:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    d["split"] = int(d["split"])
    return d


def load_done_keys(out_csv: Path):
    if not out_csv.exists():
        return set(), None
    df = pd.read_csv(out_csv)
    needed = {"fm_name", "loss", "seed", "split"}
    if not needed.issubset(df.columns):
        return set(), df
    # Backfill ft_mode for existing rows that predate the column:
    # COMPASS default was FFT, all LoRA-based models default to LoRA.
    if "ft_mode" not in df.columns:
        df["ft_mode"] = df["fm_name"].map(lambda x: "FFT" if x == "COMPASS" else "LoRA")
    done = set(
        zip(
            df["fm_name"].astype(str),
            df["loss"].astype(str),
            df["ft_mode"].astype(str),
            df["seed"].astype(int),
            df["split"].astype(int),
        )
    )
    return done, df


def _consolidate_survival(pred_batches):
    id_list, risk_list, time_list, event_list, organ_list, ct_list = [], [], [], [], [], []
    for batch in pred_batches:
        id_, risk, time, event, organ, cancer_type = batch
        id_list.extend(list(id_))
        risk_list.extend(list(risk))
        time_list.extend(list(time))
        event_list.extend(list(event))
        organ_list.extend(list(organ) if organ is not None else [None] * len(risk))
        ct_list.extend(list(cancer_type) if cancer_type is not None else [None] * len(risk))
    return id_list, risk_list, time_list, event_list, organ_list, ct_list


def eval_one_folder(model_folder: Path, dataset: str, use_gpu: bool, min_n: int, min_events: int):
    config_path = model_folder / "config.json"
    if not config_path.exists():
        return None

    config = load_json(config_path)
    pl.seed_everything(int(config.get("seed", 37)), workers=True)

    fm_name = config.get("fm_name")
    if fm_name not in FM_REGISTRY:
        raise ValueError(f"Unknown fm_name={fm_name}. Add it to FM_REGISTRY.")

    fm_cfg = config["fm_configs"][fm_name]
    ft_mode = fm_cfg.get("ft_mode", "LoRA")  # COMPASS=FFT/PFT/LFT; others=LoRA

    data_cfg = config["data"]

    # scGPT / CancerFoundation need a collate_fn that bins + pads variable-length token dicts
    collate_fn = None
    if fm_name == "scGPT":
        from functools import partial
        import torchtext
        torchtext.disable_torchtext_deprecation_warning()
        from scgpt.tokenizer import GeneVocab
        vocab = GeneVocab.from_file(os.path.join(fm_cfg["hf_model_id"], "vocab.json"))
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in vocab:
                vocab.append_token(s)
        collate_fn = partial(
            scgpt_collate_fn,
            pad_token_id=vocab[fm_cfg.get("pad_token", "<pad>")],
            pad_value=fm_cfg.get("pad_value", -2),
            max_length=fm_cfg.get("max_length", 1200),
            n_bins=fm_cfg.get("n_bins", 51),
        )
    elif fm_name == "CancerFoundation":
        import sys
        from functools import partial
        CF_DIR = os.environ.get("CANCERFOUNDATION_DIR", "/data/rbg/users/ashyamal/CancerFoundation")
        sys.path.insert(0, CF_DIR)
        from model.vocab import GeneVocab as CFGeneVocab
        vocab_file = os.path.join(CF_DIR, "model/assets/vocab.json")
        cf_vocab = CFGeneVocab.from_file(vocab_file)
        cf_vocab.set_default_index(cf_vocab["<pad>"])
        collate_fn = partial(
            cancerfoundation_collate_fn,
            pad_token_id=cf_vocab[fm_cfg.get("pad_token", "<pad>")],
            pad_value=fm_cfg.get("pad_value", -2),
            max_length=fm_cfg.get("max_length", 1200),
            n_bins=fm_cfg.get("n_bins", 51),
        )

    dm = FinetuneTokensSurvivalDataModule(
        data_path=data_cfg["data_path"],
        tokens_path=data_cfg["tokens_path"],
        split_folder=data_cfg["split_folder"],
        split_idx=int(data_cfg["split_idx"]),
        endpoint=data_cfg.get("endpoint", "PFS"),
        batch_size=data_cfg.get("batch_size", 4),
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )
    dm.prepare_data()
    dm.setup()

    ckpts = sorted([f for f in os.listdir(model_folder) if f.endswith(".ckpt")])
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt in {model_folder}")
    ckpt_path = model_folder / ckpts[0]

    ModelClass = FM_REGISTRY[fm_name]
    model = ModelClass.load_from_checkpoint(str(ckpt_path))
    model.eval()

    accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
    trainer_cfg = dict(config.get("trainer", {}))
    for key in ("devices", "accelerator", "strategy", "accumulate_grad_batches"):
        trainer_cfg.pop(key, None)

    trainer = pl.Trainer(
        default_root_dir=str(model_folder),
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
        **trainer_cfg,
    )

    with torch.no_grad():
        pred_batches = trainer.predict(model, dataloaders=dm.test_dataloader())

    _, risk, time, event, organ, cancer_type = _consolidate_survival(pred_batches)

    event_bool = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)

    results = {"overall": safe_cindex(event_bool, time, risk)}

    for ct in set(cancer_type):
        if ct is None:
            continue
        idx = [i for i, ci in enumerate(cancer_type) if ci == ct]
        if len(idx) < min_n or int(np.sum(event_bool[idx])) < min_events:
            continue
        results[str(ct)] = safe_cindex(event_bool[idx], time[idx], risk[idx])

    return results, ft_mode


def main(args):
    dataset = args.dataset
    endpoint = args.endpoint

    base = (
        REPO_ROOT / "model_outputs_finetune"
        / dataset / "survival" / endpoint
    )
    results_dir = Path(args.results_dir) if args.results_dir else base
    out_csv = Path(args.out_csv) if args.out_csv else (base / "finetune_metrics.csv")

    print(f"Dataset={dataset} endpoint={endpoint}")
    print(f"Results dir: {results_dir}")

    folders = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    print(f"Found {len(folders)} subfolders")

    done_keys, existing_df = load_done_keys(out_csv)
    print(
        f"Loaded {len(done_keys)} existing entries"
        if existing_df is not None else "No existing CSV found"
    )

    new_rows = []
    skipped = 0

    for i, folder in tqdm(enumerate(folders, 1), total=len(folders)):
        meta = parse_run_name(folder.name)
        if meta is None:
            continue
        if args.fm_name and meta["fm_name"] != args.fm_name:
            continue

        # ft_mode from dir name (new format); fall back to same logic as load_done_keys
        ft_mode_dir = meta.get("ft_mode_dir")
        if ft_mode_dir is None:
            ft_mode_dir = "FFT" if meta["fm_name"] == "COMPASS" else "LoRA"
        key = (meta["fm_name"], meta["loss"], ft_mode_dir, meta["seed"], meta["split"])
        if key in done_keys:
            skipped += 1
            if args.verbose:
                print(f"SKIP already in CSV: {key}")
            continue

        try:
            res, ft_mode = eval_one_folder(
                folder,
                dataset=dataset,
                use_gpu=True,
                min_n=args.min_n,
                min_events=args.min_events,
            )
        except Exception as e:
            row = {
                "run_dir": str(folder),
                "dataset": dataset,
                "fm_name": meta["fm_name"],
                "loss": meta["loss"],
                "ft_mode": ft_mode_dir,
                "seed": meta["seed"],
                "split": meta["split"],
                "overall": np.nan,
                "error": str(e),
            }
            new_rows.append(row)
            continue

        if res is None:
            if args.verbose:
                print(f"SKIP (no config.json): {folder.name}")
            continue

        row = {
            "run_dir": str(folder),
            "dataset": dataset,
            "fm_name": meta["fm_name"],
            "loss": meta["loss"],
            "ft_mode": ft_mode,
            "seed": meta["seed"],
            "split": meta["split"],
            "overall": res.get("overall", np.nan),
            "error": "",
        }
        for k, v in res.items():
            if k != "overall":
                row[k] = v

        new_rows.append(row)
        if args.verbose:
            print(f"[{i}/{len(folders)}] {folder.name} overall={row['overall']:.4f}")

    print(f"Skipped {skipped} runs already in CSV")

    df_new = pd.DataFrame(new_rows)
    df_out = (
        df_new if existing_df is None
        else pd.concat([existing_df, df_new], ignore_index=True, sort=False)
    )

    base_cols = ["dataset", "fm_name", "loss", "ft_mode", "seed", "split", "overall"]
    extra_cols = ["run_dir", "error"]
    other_cols = sorted([c for c in df_out.columns if c not in set(base_cols + extra_cols)])
    cols = base_cols + other_cols + extra_cols
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} rows to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["TCGA", "CPTAC", "CALGB", "Hartwig", "ICGC", "TARGET_AML"])
    p.add_argument("--endpoint", default="PFS")
    p.add_argument("--fm_name", default=None,
                   help="Only evaluate folders for this FM (e.g. BulkFormer, CancerFoundation). "
                        "If omitted, evaluate all FMs found in results_dir.")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--min_n", type=int, default=0)
    p.add_argument("--min_events", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)

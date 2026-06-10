"""
train/train_compass_ICI.py

Fine-tune (or zero-shot evaluate) COMPASS on ICI response cohorts.

Paradigms  : within_cohort | loco | transfer
Modes      : LFT | PFT | FFT | NFT
             NFT = zero-shot prototype prediction (no gradient updates)

Usage:
  python -m train.train_compass_ICI \\
      --paradigm loco --cohort_tag held_out_Riaz \\
      --split 0 --mode PFT --seed 42

  python -m train.train_compass_ICI \\
      --paradigm within_cohort --cohort_tag Hugo \\
      --split 0 --mode FFT --seed 37
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from compass import loadcompass, FineTuner

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[1]
COMPASS_PT    = REPO_ROOT / "data" / "COMPASS" / "pretrainer.pt"
GENE_MAP_CSV  = REPO_ROOT / "data" / "COMPASS" / "compass_gene_map.csv"
ICI_META_JSON = REPO_ROOT / "data" / "RNA_embeddings" / "ICI" / "ICI_meta.json"
ICI_TPM_PQ    = REPO_ROOT / "data" / "RNA_embeddings" / "ICI" / "ICI_tpm.parquet"
SPLITS_ROOT   = REPO_ROOT / "splits" / "ICI"
OUTPUTS_ROOT  = REPO_ROOT / "model_outputs" / "ICI" / "compass"

# COMPASS cancer type → integer index
CANCER_MAPPING = {
    "ACC":0,"BLCA":1,"BRCA":2,"CESC":3,"CHOL":4,"COAD":5,"DLBC":6,"ESCA":7,
    "GBM":8,"HNSC":9,"KICH":10,"KIRC":11,"KIRP":12,"LAML":13,"LGG":14,"LIHC":15,
    "LUAD":16,"LUSC":17,"MESO":18,"OV":19,"PAAD":20,"PCPG":21,"PRAD":22,"READ":23,
    "SARC":24,"SKCM":25,"STAD":26,"TGCT":27,"THCA":28,"THYM":29,"UCEC":30,"UCS":31,
    "UVM":32,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tpm_map():
    """
    Returns dict: sample_id -> pd.Series(tpm, index=gene_symbols).
    Loads from ICI_tpm.parquet (pre-exported with correct IDs via rna_modeling env).
    """
    tpm_df = pd.read_parquet(ICI_TPM_PQ)
    return {sid: tpm_df.loc[sid] for sid in tpm_df.index}


def load_compass_genes():
    """Returns list of gene symbols in COMPASS vocabulary order."""
    gmap = pd.read_csv(GENE_MAP_CSV)
    return gmap["gene_name"].tolist()


def load_meta():
    """Returns dict: sample_id -> {response, cancer_type, cohort}."""
    with open(ICI_META_JSON) as f:
        return json.load(f)


def load_split(paradigm: str, cohort_tag: str, split_idx: int):
    path = SPLITS_ROOT / paradigm / cohort_tag / f"split_{split_idx}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)  # {"train": [...], "val": [...], "test": [...]}


# ── DataFrame builders ────────────────────────────────────────────────────────

def build_dfcx(sample_ids, tpm_map, meta, compass_genes, split_tag=""):
    """
    Build COMPASS input DataFrame:
      rows = samples, columns = [cancer_code] + [gene1, gene2, ...]
    Missing genes filled with 0.
    """
    rows = []
    valid_ids = []
    for sid in sample_ids:
        if sid not in tpm_map:
            continue
        tpm = tpm_map[sid]
        # align to COMPASS genes; fill missing with 0
        aligned = tpm.reindex(compass_genes, fill_value=0.0).fillna(0.0)
        cancer_code = CANCER_MAPPING.get(meta[sid]["cancer_type"], 0)
        row = pd.Series({"cancer_code": float(cancer_code), **aligned.to_dict()})
        rows.append(row)
        valid_ids.append(sid)
    df = pd.DataFrame(rows, index=valid_ids)

    # Sanity check: report zero-filled genes (missing from cohort or NaN-filled)
    gene_cols = [c for c in df.columns if c != "cancer_code"]
    zero_genes = (df[gene_cols] == 0.0).all(axis=0).sum()
    print(f"  [{split_tag}] {len(df)} samples | {len(gene_cols)} COMPASS genes | "
          f"{zero_genes} genes all-zero across samples")

    return df, valid_ids


def build_dfy(sample_ids, meta):
    """
    Build label DataFrame: one-hot encoded, shape (N, 2).
    Column 0 = NR (non-responder), column 1 = R (responder).
    """
    labels = [meta[sid]["response"] for sid in sample_ids]
    onehot = np.zeros((len(labels), 2), dtype=np.float32)
    for i, y in enumerate(labels):
        onehot[i, int(y)] = 1.0
    return pd.DataFrame(onehot, index=sample_ids, columns=["NR", "R"])


# ── Run ───────────────────────────────────────────────────────────────────────

def run(args):
    print(f"Mode={args.mode}  paradigm={args.paradigm}  tag={args.cohort_tag}  "
          f"split={args.split}  seed={args.seed}")

    # Load shared resources
    print("Loading data...")
    compass_genes = load_compass_genes()
    tpm_map       = load_tpm_map()
    meta          = load_meta()

    split = load_split(args.paradigm, args.cohort_tag, args.split)
    train_ids, val_ids, test_ids = split["train"], split["val"], split["test"]

    # Build DataFrames
    dfcx_train, train_ids = build_dfcx(train_ids, tpm_map, meta, compass_genes, "train")
    dfcx_val,   val_ids   = build_dfcx(val_ids,   tpm_map, meta, compass_genes, "val")
    dfcx_test,  test_ids  = build_dfcx(test_ids,  tpm_map, meta, compass_genes, "test")

    dfy_train = build_dfy(train_ids, meta)
    dfy_val   = build_dfy(val_ids,   meta)
    dfy_test  = build_dfy(test_ids,  meta)

    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # Output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"COMPASS_{args.mode}_seed_{args.seed}_split_{args.split}_{ts}"
    out_dir  = OUTPUTS_ROOT / args.paradigm / args.cohort_tag / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    print(f"Loading COMPASS from {COMPASS_PT}")
    pretrainer = loadcompass(str(COMPASS_PT))

    # COMPASS uses drop_last=True internally; cap batch size to avoid empty DataLoader
    effective_batch_size = min(args.batch_size, len(train_ids))
    if effective_batch_size < args.batch_size:
        print(f"  Warning: train set ({len(train_ids)}) < batch_size ({args.batch_size}), "
              f"using batch_size={effective_batch_size}")

    wandb_run_name = f"{args.paradigm}/{args.cohort_tag}/{args.mode}/split{args.split}"

    shared_ft_kwargs = dict(
        device=args.device,
        seed=args.seed,
        task_dim=2,
        task_type="c",
        task_name="ici_response",
        task_class_weight=[1, 2],
        work_dir=str(out_dir),
        save_best_model=False,
        verbose=args.verbose,
        with_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # ── Fine-tune or zero-shot predict ────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity="ashyamal-mit",
            name=wandb_run_name,
            config={
                "paradigm":   args.paradigm,
                "cohort_tag": args.cohort_tag,
                "mode":       args.mode,
                "split":      args.split,
                "seed":       args.seed,
                "lr":         args.lr,
                "max_epochs": args.max_epochs,
                "patience":   args.patience,
                "batch_size": args.batch_size,
            },
            reinit=True,
        )

    ft = FineTuner(
            pretrainer,
            mode=args.mode,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            task_dense_layer=[16],
            task_batch_norms=True,
            batch_size=effective_batch_size,
            **shared_ft_kwargs,
        )
    ft.tune_with_test(dfcx_train, dfy_train, dfcx_val, dfy_val)

    # ── Predictions ───────────────────────────────────────────────────────────
    print("Predicting on test set...")
    _, dfp_test = ft.predict(dfcx_test)
    dfp_test.index = test_ids

    print("Predicting on val set...")
    _, dfp_val = ft.predict(dfcx_val)
    dfp_val.index = val_ids

    # dfp has shape (N, 2) with class probabilities; column 1 = P(responder)
    prob_col = dfp_test.columns[1]  # positive-class column

    def make_pred_df(ids, dfp, split_name):
        df = pd.DataFrame({
            "sample_id":   ids,
            "score":       dfp[prob_col].values,
            "response":    [meta[s]["response"]    for s in ids],
            "cohort":      [meta[s]["cohort"]      for s in ids],
            "cancer_type": [meta[s]["cancer_type"] for s in ids],
            "split":       split_name,
        })
        return df

    pred_test = make_pred_df(test_ids, dfp_test, "test")
    pred_val  = make_pred_df(val_ids,  dfp_val,  "val")
    preds     = pd.concat([pred_test, pred_val], ignore_index=True)

    preds.to_csv(out_dir / "predictions.csv", index=False)
    print(f"Saved predictions → {out_dir / 'predictions.csv'}")

    # Save config
    config = {
        "paradigm":   args.paradigm,
        "cohort_tag": args.cohort_tag,
        "mode":       args.mode,
        "seed":       args.seed,
        "split":      args.split,
        "train_n":    len(train_ids),
        "val_n":      len(val_ids),
        "test_n":     len(test_ids),
        "lr":         args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience":   args.patience,
        "batch_size": args.batch_size,
        "run_dir":    str(out_dir),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config → {out_dir / 'config.json'}")

    if args.wandb:
        import wandb
        # Log final test metrics to wandb summary
        from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
        test_sub = preds[preds["split"] == "test"]
        try:
            wandb.summary["test_auroc"]  = roc_auc_score(test_sub["response"], test_sub["score"])
            wandb.summary["test_pr_auc"] = average_precision_score(test_sub["response"], test_sub["score"])
            wandb.summary["test_mcc"]    = matthews_corrcoef(test_sub["response"], (test_sub["score"] >= 0.5).astype(int))
        except Exception:
            pass
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm",   required=True,
                   choices=["within_cohort", "loco", "transfer"])
    p.add_argument("--cohort_tag", required=True, type=str,
                   help="e.g. 'held_out_Riaz', 'Hugo', 'Hugo_to_Rose'")
    p.add_argument("--split",      type=int, required=True, help="0–4")
    p.add_argument("--mode",       default="PFT",
                   choices=["LFT", "PFT", "FFT"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int,   default=500)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--wandb",         action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", type=str, default="compass-ICI")
    args = p.parse_args()
    run(args)

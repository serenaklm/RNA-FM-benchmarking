# Simple Baselines Outperform Transcriptomic Foundation Model Embeddings for Cancer Prognostic Modeling

This repository contains the code for the paper **"Simple Baselines Outperform Transcriptomic Foundation Model Embeddings for Cancer Prognostic Modeling"** by Ananth Shyamal, Ling Min Serena Khoo, and Regina Barzilay (MIT CSAIL).

We benchmarked six state-of-the-art bulk and single-cell transcriptomic foundation models (FMs) against simple baseline dimensionality reduction methods for predicting cancer patient progression-free survival (PFS) from bulk tumor RNA-seq profiles. Across four diverse cancer cohorts (TCGA, Hartwig, CPTAC, TARGET-AML), simple baseline methods consistently outperformed all six foundation models.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
  - [Dataset-Specific Preprocessing](#dataset-specific-preprocessing)
  - [FM Embedding Extraction](#fm-embedding-extraction)
  - [Baseline Embedding Computation](#baseline-embedding-computation)
  - [Data Splits](#data-splits)
- [Training](#training)
  - [Main Benchmarking (Frozen Embeddings)](#main-benchmarking-frozen-embeddings)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Clinical Baselines](#clinical-baselines)
  - [Concatenation Experiment (FM + HVG)](#concatenation-experiment-fm--hvg)
  - [Fine-Tuning Experiment](#fine-tuning-experiment)
  - [Auxiliary Experiments](#auxiliary-experiments)
- [Evaluation](#evaluation)
- [Analysis](#analysis)
- [Model Architecture & Loss Functions](#model-architecture--loss-functions)
- [Configuration](#configuration)
- [Outputs & Results](#outputs--results)
- [Contact](#contact)

---

## Overview

The benchmarking pipeline embeds bulk RNA-seq profiles using either transcriptomic foundation models or baseline dimensionality reduction methods, then trains a survival predictor (linear network or residual MLP) on top of the frozen embeddings to predict PFS. Performance is evaluated using Harrell's concordance index (C-index).

**Foundation models evaluated:**
- Bulk: BulkRNABert, BulkFormer, COMPASS
- Single-cell: scGPT, STATE-SE, CancerFoundation

**Baselines evaluated:**
- Highly variable genes (HVGs): k ∈ {18, 50, 132, 500, 729, 1250, 5000, 10000}
- Gene-set-based methods (max pooling and ssGSEA): Hallmark, Immuno, AURORA, Combined gene sets
- PCA: 50 and 100 components
- Raw expression (no dimensionality reduction)
- Clinical covariates (age, gender, stage, tumor location) via Cox PH

**Cohorts:** TCGA (n=10,388), Hartwig (n=2,153), CPTAC (n=1,222), TARGET-AML (n=1,902)

---

## Repository Structure

```
RNA-FM-benchmarking/
│
├── configs/                          # YAML configuration files
│   ├── all_models.yaml               # Main config for frozen-embedding experiments
│   ├── all_models_HVG_pooling.yaml   # Config for HVG-pooled FM embedding experiments
│   └── finetune.yaml                 # Config for fine-tuning experiments (LoRA / full FT)
│
├── data/                             # Processed data (not included; see Datasets section)
│   ├── RNA_embeddings/               # Processed per-dataset .pkl files with all embeddings
│   └── tokens/                       # Pre-tokenized inputs for fine-tuning
│
├── splits/                           # Train/val/test split indices per dataset
│   ├── TCGA_cleaned/
│   ├── Hartwig/
│   ├── CPTAC/
│   └── TARGET_AML/
│
├── data_preprocessing/               # All data processing scripts
│   ├── preprocess_utils.py           # Shared preprocessing utility functions
│   ├── get_data_splits.py            # Generate stratified train/val/test splits
│   ├── compute_PCA.py                # Compute and save PCA embeddings
│   ├── get_ssGSEA_scores.py          # Compute ssGSEA gene-set scores
│   ├── get_genes_sets.py             # Load and manage gene set collections
│   ├── dict_to_gmt.py                # Convert gene set dicts to GMT format
│   ├── get_gene_length.py            # Utility for gene length annotation
│   ├── TCGA/                         # TCGA-specific preprocessing
│   │   ├── get_TCGA_embeddings.py    # Assemble TCGA processed .pkl with all embeddings
│   │   ├── augment_TCGA_with_HVG_embeddings.py
│   │   ├── process_tcga.ipynb        # Notebook: raw TCGA → h5ad
│   │   └── process_cleaned_TCGA_clinical.ipynb
│   ├── Hartwig/                      # Hartwig-specific preprocessing
│   │   ├── get_Hartwig_embeddings.py
│   │   ├── get_Hartwig_RNA.py
│   │   └── preprocess_Hartwig.ipynb
│   ├── CPTAC/                        # CPTAC-specific preprocessing
│   │   ├── get_CPTAC_embeddings.py
│   │   └── preprocess_CPTAC.ipynb
│   ├── TARGET_AML/                   # TARGET-AML-specific preprocessing
│   │   ├── get_TARGET_AML_embeddings.py
│   │   └── preprocess_TARGET_AML.ipynb
│   ├── get_FM_embeddings/            # Scripts to extract FM embeddings
│   │   ├── BulkRNABert.py
│   │   ├── BulkFormer.py (via BulkFormer_HVG_pooling.py)
│   │   ├── COMPASS.py
│   │   ├── scGPT.py
│   │   ├── STATE.py
│   │   └── CancerFoundation (via official_bulkrnabert_embedding_extraction.py)
│   └── tokenize_for_finetuning/      # Pre-tokenize data for fine-tuning
│       ├── BulkRNABert_tokenize.py
│       ├── BulkFormer_tokenize.py
│       ├── scGPT_tokenize.py
│       ├── STATE_tokenize.py
│       ├── CancerFoundation_tokenize.py
│       └── COMPASS_preprocess.py
│
├── scripts/                          # Shell scripts for running experiments
│   ├── embeddings/                   # Embedding extraction orchestration
│   │   ├── run_all_embeddings.sh     # Run all embedding types for a dataset
│   │   ├── run_BulkRNABert.sh
│   │   ├── run_BulkFormer.sh
│   │   ├── run_COMPASS.sh
│   │   ├── run_scGPT.sh
│   │   ├── run_STATE.sh
│   │   ├── run_CancerFoundation.sh
│   │   ├── run_ssGSEA.sh
│   │   └── get_data_preprocess_stats.sh
│   ├── train_RNA_models.sh           # Main training script (frozen embeddings)
│   ├── tune_RNA_models.sh            # Hyperparameter tuning via Optuna
│   ├── train_RNA_models_clinical.sh  # Train clinical covariate Cox PH baselines
│   ├── train_RNA_models_concat.sh    # Train FM + HVG concatenation models
│   ├── train_TCGA_HVG_pooling.sh     # Train on HVG-pooled FM embeddings
│   ├── train_binned_HVG_models.sh    # Train binned HVG ablation models
│   ├── resume_binned_HVG_models.sh   # Resume partial binned HVG runs
│   ├── train_finetune_BulkRNABert.sh # Fine-tune BulkRNABert (LoRA)
│   ├── train_finetune_BulkFormer.sh  # Fine-tune BulkFormer (LoRA)
│   ├── train_finetune_scGPT.sh       # Fine-tune scGPT (LoRA)
│   ├── train_finetune_STATE.sh       # Fine-tune STATE-SE (LoRA)
│   ├── train_finetune_COMPASS.sh     # Fine-tune COMPASS
│   ├── train_finetune_CancerFoundation.sh
│   ├── run_all_single_dataset.sh     # Launch all experiments for one dataset
│   ├── run_eval_all_datasets.sh      # Evaluate all models across all datasets
│   ├── run_eval_all_datasets_concat.sh
│   ├── run_eval_all_datasets_clinical.sh
│   ├── run_eval_binned_HVG.sh
│   ├── run_compute_val_cindex.sh     # Compute validation C-indices
│   ├── eval_finetune.sh              # Evaluate fine-tuned FM models
│   └── run_linear_probing.sh         # Run linear probing experiment
│
├── train/                            # Python training modules
│   ├── train_RNA_model.py            # Main training script for frozen-embedding models
│   ├── train_RNA_model_HP_tune.py    # Single-trial training for Optuna
│   ├── tune.py                       # Optuna HP tuning entry point
│   ├── agg_best_hps.py               # Aggregate per-split HP results into one JSON
│   ├── train_finetune.py             # Fine-tune a foundation model end-to-end
│   ├── train_RNA_model_clinical.py   # Train clinical-only Cox PH baseline
│   ├── train_simple_clinical_baseline.py
│   ├── train_clinical_baseline.py    # Extended Cox PH with all clinical features
│   ├── train_RNA_model_concat.py     # Train FM + HVG concatenated models
│   ├── train_RNA_model_on_pooled_HVG_embeds.py  # Train on HVG-pooled FM embeddings
│   └── train_binned_HVG_model.py     # Train binned HVG ablation models
│
├── val/
│   └── compute_val_cindex.py         # Compute validation C-indices across all runs
│
├── test/                             # Evaluation modules
│   ├── eval_all_RNA_models.py        # Evaluate all frozen-embedding models
│   ├── eval_all_RNA_models_concat.py # Evaluate concatenation models
│   ├── eval_all_RNA_models_clinical.py
│   ├── eval_all_RNA_pooled_HVG.py    # Evaluate HVG-pooled FM models
│   ├── eval_binned_HVG_models.py     # Evaluate binned HVG ablation models
│   └── eval_finetune.py              # Evaluate fine-tuned FM models
│
├── pytorch_modules/                  # Core PyTorch model library
│   ├── data/
│   │   ├── data.py                   # DataModules for all datasets (survival/response)
│   │   ├── finetune_data.py          # DataModules for fine-tuning (token-based)
│   │   └── binned_data.py            # DataModule for binned HVG ablation
│   ├── models/
│   │   ├── survival_models.py        # RNA_SurvivalModel and RNA_ResponseModel (Lightning)
│   │   └── finetune_models.py        # FM_REGISTRY: LoRA-wrapped FM models for fine-tuning
│   ├── losses/
│   │   ├── cindex_loss.py            # Pairwise ranking loss (logistic C-index approximation)
│   │   ├── cox_loss.py               # Cox proportional hazards partial log-likelihood
│   │   └── elastic_loss.py           # Elastic net regularization wrapper
│   ├── nn/
│   │   ├── residual_MLP.py           # 3-layer residual MLP architecture
│   │   └── MLP.py                    # Standard MLP architecture
│   └── utils/
│       ├── utils.py                  # General utilities (config loading, pickle I/O, etc.)
│       └── split_data.py             # Data splitting utilities
│
├── analysis/
│   ├── extract_linear_weights.py     # Extract and rank gene weights from Linear CIndex models
│   └── linear_gene_weights.pkl       # Saved gene weight outputs
│
├── notebooks/
│   ├── analyze_results.ipynb         # Main results analysis and figure generation
│   ├── analyze_hp_results.ipynb      # Hyperparameter tuning analysis
│   ├── analyze_datasets.ipynb        # Dataset statistics and EDA
│   └── analyze_clinical_metadata.ipynb
│
├── hp_results/                       # Hyperparameter tuning outputs
│   └── all_aggregated_hps.json       # Aggregated best HPs per embedding/arch/loss
│
├── model_outputs/                    # Frozen-embedding model checkpoints and metrics
├── model_outputs_finetune/           # Fine-tuned model checkpoints and metrics
├── concat_model_outputs/             # FM + HVG concatenation model outputs
├── benchmarking_results/             # Aggregated evaluation CSVs
└── logs/                             # Training and evaluation logs
```

---

## Datasets

Four cancer cohorts are used in this study:

| Cohort | Patients | Samples | Cancer Type | Source |
|--------|----------|---------|-------------|--------|
| TCGA | 10,388 | 10,388 | Primary pan-cancer (33 types) | NCI GDC |
| Hartwig | 2,153 | 2,232 | Metastatic pan-cancer | Hartwig Medical Foundation |
| CPTAC | 1,222 | 1,222 | Primary pan-cancer | NCI GDC |
| TARGET-AML | 1,902 | 2,586 | Pediatric AML | Xena Browser / NCI GDC |

All datasets use RNA-seq data in TPM format. Genes with mean TPM < 0.1 across all samples were filtered out. Only protein-coding genes were retained for baseline methods; foundation models received all post-filter genes.

### Downloading the Data

- **TCGA & CPTAC**: Download from the [NCI GDC Data Portal](https://portal.gdc.cancer.gov/). Preprocessed TPM RNA-seq is queried via the GDC API. TCGA PFS labels are from the [TCGA Pan-Cancer Clinical Data Resource](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018).
- **Hartwig**: Submit a data access request to the [Hartwig Medical Foundation](https://www.hartwigmedicalfoundation.nl/en/data/data-access-request/). Uses Isofox pipeline output.
- **TARGET-AML**: TPM counts from the [Xena Browser](https://xenabrowser.net/); clinical data including PFS follow-up from the [GDC Portal](https://portal.gdc.cancer.gov/projects/TARGET-AML).

---

## Setup

Each foundation model may require its own conda environment due to dependency conflicts. The core benchmarking environment is:

```bash
conda create -n rna_modeling python=3.10
conda activate rna_modeling
pip install -r requirements.txt
```

The fine-tuning and embedding extraction scripts for some FMs use model-specific environments:

| Model | Environment |
|-------|-------------|
| BulkRNABert, COMPASS | `rna_modeling` |
| BulkFormer | `bulkformer` |
| CancerFoundation | `cancerfoundation` |
| scGPT | `scgpt_env` |
| STATE-SE | `state` |

All scripts are run from the **repository root** as Python modules (e.g., `python -m train.train_RNA_model`).

---

## Data Preprocessing

### Dataset-Specific Preprocessing

Each dataset has a dedicated preprocessing notebook and embedding assembly script under `data_preprocessing/<DATASET>/`.

**Steps (run from the dataset subdirectory or as indicated):**

1. **TCGA**: `data_preprocessing/TCGA/process_tcga.ipynb` — converts raw GDC downloads to an h5ad file. `process_cleaned_TCGA_clinical.ipynb` handles PFS label cleaning from the Pan-Cancer Clinical Data Resource.

2. **Hartwig**: `data_preprocessing/Hartwig/preprocess_Hartwig.ipynb` — aligns Isofox TPM output with clinical data to compute PFS.

3. **CPTAC**: `data_preprocessing/CPTAC/preprocess_CPTAC.ipynb`

4. **TARGET-AML**: `data_preprocessing/TARGET_AML/preprocess_TARGET_AML.ipynb`

After preprocessing, the per-dataset embedding assembly scripts consolidate all embeddings (FM embeddings, ssGSEA scores, PCA projections, raw expression, clinical features) into a single `.pkl` file per dataset used by the DataModules at training time:

```bash
python -m data_preprocessing.TCGA.get_TCGA_embeddings
python -m data_preprocessing.Hartwig.get_Hartwig_embeddings
python -m data_preprocessing.CPTAC.get_CPTAC_embeddings
python -m data_preprocessing.TARGET_AML.get_TARGET_AML_embeddings
```

The output `.pkl` files are stored in `data/RNA_embeddings/` and are the primary data inputs for all training scripts.

### FM Embedding Extraction

Foundation model embeddings are extracted using the scripts in `data_preprocessing/get_FM_embeddings/`. A convenience wrapper runs all embedding types for a single dataset:

```bash
bash scripts/embeddings/run_all_embeddings.sh --dataset TCGA --gpu 0
```

Individual FM embedding scripts can also be run directly:

```bash
bash scripts/embeddings/run_BulkRNABert.sh --dataset TCGA --gpu 0
bash scripts/embeddings/run_BulkFormer.sh  --dataset TCGA --gpu 0
bash scripts/embeddings/run_COMPASS.sh     --dataset TCGA --gpu 0
bash scripts/embeddings/run_scGPT.sh       --dataset TCGA --gpu 0
bash scripts/embeddings/run_STATE.sh       --dataset TCGA
bash scripts/embeddings/run_CancerFoundation.sh --dataset TCGA --gpu 0
```

**Notes:**
- Single-cell FMs (scGPT, STATE-SE, CancerFoundation) require prior HVG selection (top 5,000 HVGs for scGPT/STATE; top 1,199 for CancerFoundation).
- Bulk FMs (BulkRNABert, BulkFormer) use all shared genes with model-specific normalization (see paper Table 6).
- COMPASS additionally requires cancer-type annotations (TCGA project ID format).
- BulkRNABert embedding extraction is slow on CPU (~150 samples/hour); the `run_all_embeddings.sh` script omits it from the GPU pipeline by default.

### Baseline Embedding Computation

**PCA:**
```bash
cd data_preprocessing
python compute_PCA.py --data_set TCGA
```
PCA is fit on training samples per split and projections (50 and 100 components) are stored per sample.

**ssGSEA gene-set scores:**
```bash
bash scripts/embeddings/run_ssGSEA.sh --dataset TCGA
# or directly:
cd data_preprocessing
python get_ssGSEA_scores.py --gene_set all --data_set TCGA
```
Gene-set collections used: Hallmark (50 sets), Immuno (132 sets from COMPASS developers), AURORA (729 sets), and Combined (910 sets after deduplication). Max pooling scores are computed inline during the embedding assembly step.

### Data Splits

Five independent 70/15/15% train/val/test splits are created per dataset at the patient level:

```bash
python -m data_preprocessing.get_data_splits
```

Splits are stratified by cancer type (TCGA), organ group (CPTAC), or cancer type × treatment response (Hartwig). For Hartwig and TARGET-AML (multi-sample patients), all samples from a patient are assigned to the same split to prevent data leakage. Split indices are saved as `.pkl` files in `splits/<DATASET>/split_0.pkl` through `split_4.pkl`.

---

## Training

All training scripts are run as Python modules from the **repository root**. Models are logged to Weights & Biases and checkpoints are saved locally.

### Main Benchmarking (Frozen Embeddings)

The primary training script is `train/train_RNA_model.py`. It supports all embedding types, both architectures (Linear, ResidualMLP), and both loss functions (Cox, CIndex).

**Using the shell launcher** (recommended):

```bash
# Usage: ./scripts/train_RNA_models.sh <GPU_ID> <DATASET> <TASK> <EXPERIMENT> <ARCH> <LOSS_NAME>
#
# EXPERIMENT options:
#   HVG        — all 8 HVG sizes (18, 50, 132, 500, 729, 1250, 5000, 10000)
#   HVG1/2/3   — subsets of HVG sizes (for parallel execution)
#   GS         — all gene-set embeddings (4 collections × 2 pooling strategies)
#   SSGSEA     — ssGSEA pooling only
#   MAX        — max pooling only
#   HALLMARK   — Hallmark gene set only
#   FM_BULK    — BulkFormer, BulkRNABert, COMPASS
#   FM_SC      — scGPT, STATE, CancerFoundation
#   PCA        — PCA with 50 and 100 components
#   BASELINE   — raw expression (all genes, no dimensionality reduction)
#   ALL        — all of the above

# Examples:
bash scripts/train_RNA_models.sh 0 TCGA   survival HVG      ResidualMLP CIndex
bash scripts/train_RNA_models.sh 1 TCGA   survival FM_BULK  ResidualMLP Cox
bash scripts/train_RNA_models.sh 2 Hartwig survival GS       Linear      CIndex
bash scripts/train_RNA_models.sh 3 CPTAC  survival FM_SC    ResidualMLP CIndex
```

To run all experiments for a single dataset across multiple GPUs in parallel:

```bash
bash scripts/run_all_single_dataset.sh TCGA
```

This distributes experiments across GPUs 3, 5, 6, 7 and sweeps both seeds (37 and 42).

**Running the Python script directly:**

```bash
python -m train.train_RNA_model \
  --config_path configs/all_models.yaml \
  --dataset TCGA \
  --task survival \
  --arch ResidualMLP \
  --loss_name CIndex \
  --rna_embedding_key BulkFormer \
  --seed 37 \
  --split_idx 0 \
  --hp_path hp_results/all_aggregated_hps.json
```

Key arguments:
- `--rna_embedding_key`: embedding to use. Options include `BulkFormer`, `BulkRNABert`, `COMPASS`, `scGPT`, `STATE`, `CancerFoundation`, `RNA_seq` (with `--n_HVGs` for HVG selection), `PCA_50`, `PCA_100`, `ssGSEA_<collection>`, `max_<collection>`
- `--n_HVGs`: number of HVGs when `rna_embedding_key=RNA_seq` (e.g., 1250, 5000, 10000)
- `--arch`: `Linear` | `MLP` | `ResidualMLP`
- `--loss_name`: `Cox` | `CIndex`
- `--hp_path`: path to tuned hyperparameter JSON (produced by HP tuning step)

### Hyperparameter Tuning

Hyperparameters were tuned on TCGA using Optuna's Tree-structured Parzen Estimator (TPE) with 50 trials per configuration per split. Tuning is done for each embedding × architecture × loss combination.

```bash
# Usage: ./scripts/tune_RNA_models.sh <RNA_EMBEDDING_KEY> <ARCH> <LOSS_NAME>
# Runs 5 splits in parallel across 2 GPUs.

bash scripts/tune_RNA_models.sh BulkFormer    ResidualMLP CIndex
bash scripts/tune_RNA_models.sh RNA_seq_1250  Linear      Cox
bash scripts/tune_RNA_models.sh ssGSEA_combined ResidualMLP CIndex
```

Tuning results are written to `hp_results/` (one JSON per split). Aggregate the per-split results into a single file with:

```bash
python -m train.agg_best_hps
```

This produces `hp_results/all_aggregated_hps.json`, which maps `<embedding_key>_<arch>_<loss>` → best hyperparameters. The same tuned HPs are reused across all datasets.

**Hyperparameter search spaces:**

| Hyperparameter | Range | Distribution |
|----------------|-------|--------------|
| Learning rate | [1e-5, 1e-3] | Log-uniform |
| Weight decay | [1e-7, 1e-4] | Log-uniform |
| Penalty weight | [1e-4, 1e-2] | Log-uniform |
| Dropout rate | [0.1, 0.5] | Uniform (ResidualMLP only) |

### Clinical Baselines

Cox proportional hazards models trained on clinical covariates (age, gender, stage, tumor location):

```bash
# Simple clinical baseline (age, gender, stage, location only):
python -m train.train_simple_clinical_baseline

# Extended clinical baseline (all available clinical features with missingness encoding):
python -m train.train_clinical_baseline

# Via shell script (runs all 4 datasets):
bash scripts/train_RNA_models_clinical.sh
```

Clinical features per dataset are documented in the paper (Table 8). Patients with missing features are excluded from the simple baseline; the extended baseline (`train_clinical_baseline.py`) explicitly models missingness.

### Concatenation Experiment A — FM + HVG

Tests whether concatenating FM embeddings with the best-performing HVG baseline adds incremental value over either modality alone.

The model input is the FM embedding and the log-normalized HVG expression vector scaled independently and then concatenated, passed to the same Linear / ResidualMLP + Cox / CIndex setup used in main benchmarking. The best-performing HVG count per dataset is used: TCGA: 10,000; Hartwig: 5,000; CPTAC & TARGET-AML: 1,250.

```bash
# Runs all 4 datasets across 4 GPUs in parallel.
# 2 archs × 2 losses × 6 FMs × 2 seeds × 5 splits = 120 runs per GPU.
bash scripts/train_RNA_models_concat.sh
```

Or a single configuration:

```bash
python -m train.train_RNA_model_concat \
  --config_path configs/all_models.yaml \
  --dataset TCGA \
  --arch ResidualMLP \
  --loss_name CIndex \
  --fm_embedding_key BulkFormer \
  --n_HVGs 10000 \
  --seed 37 \
  --split_idx 0 \
  --hp_path hp_results/all_aggregated_hps.json
```

Outputs are saved to `concat_model_outputs/`. Evaluate with:

```bash
bash scripts/run_eval_all_datasets_concat.sh
# or: python -m test.eval_all_RNA_models_concat --dataset TCGA --endpoint PFS
```

### Concatenation Experiment B — FM + Clinical Features

Tests whether appending encoded clinical metadata to a frozen FM embedding improves PFS prediction. This is implemented in `train/train_RNA_model_clinical.py` and uses a dedicated `ClinicalConcatSurvivalDataModule` that:

1. Loads the FM (or raw RNA_seq) embedding for each patient from the standard `.pkl` file.
2. Encodes clinical features from the raw clinical TSV **fit on training data only**, preventing leakage. The encoding strategy per feature type is:
   - **Continuous** (e.g., age): z-score normalization + a binary `__missing` indicator flag
   - **Ordinal** (e.g., AJCC stage): integer map (0–4) + `__missing` flag
   - **Binary** (e.g., prior malignancy): {0, 1} (0.5 = unknown) + `__missing` flag
   - **Categorical** (e.g., cancer type, race): one-hot with an explicit MISSING category; rare values (< 10 training samples) are grouped into OTHER
3. Concatenates `[FM_embedding | clinical_features]` as the model input vector.

Clinical features used per dataset:

| Dataset | Features |
|---------|----------|
| TCGA | Project ID (cancer type), age, gender, race, AJCC stage (overall + T/N/M), prior malignancy, prior treatment, synchronous malignancy, tissue of origin, grade, residual disease, morphology, primary diagnosis |
| CPTAC | Age, gender, race, AJCC stage (overall + T/N/M), tissue of origin, grade, residual disease, morphology, primary diagnosis |
| Hartwig | Age at biopsy, gender, primary tumor location, pre-treatment (systemic), pre-treatment (radiotherapy) |
| TARGET-AML | Age, gender, race, ethnicity |

Embedding keys evaluated: all 6 FMs plus `RNA_seq` (raw expression).

```bash
# Runs all 4 datasets × 2 archs across 8 GPUs in parallel.
# 2 losses × 7 FM keys × 2 seeds × 5 splits = 140 runs per GPU.
bash scripts/train_RNA_models_clinical.sh
```

Or a single configuration:

```bash
python -m train.train_RNA_model_clinical \
  --config_path configs/all_models.yaml \
  --dataset TCGA \
  --arch ResidualMLP \
  --loss_name CIndex \
  --rna_embedding_key BulkFormer \
  --seed 37 \
  --split_idx 0 \
  --hp_path hp_results/all_aggregated_hps.json \
  --clinical_base /data/rbg/users/ashyamal/RNA-FM-benchmarking
```

Outputs are saved to `clinical_concat_model_outputs/`. Evaluate with:

```bash
bash scripts/run_eval_all_datasets_clinical.sh
# or: python -m test.eval_all_RNA_models_clinical --dataset TCGA --endpoint PFS
```

### Fine-Tuning Experiment

Fine-tunes foundation model parameters end-to-end alongside a survival head, rather than using frozen embeddings. Transformer-based FMs use LoRA (Low-Rank Adaptation); COMPASS uses partial fine-tuning. Each FM requires its own conda environment (see Setup). The entry point is `train/train_finetune.py`.

#### Step 1 — Pre-tokenize data

Each FM has a dedicated tokenization script in `data_preprocessing/tokenize_for_finetuning/` that converts the raw h5ad expression data into model-specific token arrays and saves them as a single `.pkl` file: `{patient_id: np.array(...)}`. This needs to run once per FM per dataset before any fine-tuning.

```bash
cd data_preprocessing/tokenize_for_finetuning

# BulkRNABert: bins each gene's log10(1+TPM) value into 64 linear bins.
# Produces np.int8 arrays of shape (n_genes,).
python BulkRNABert_tokenize.py --data_set TCGA
python BulkRNABert_tokenize.py --data_set CPTAC
python BulkRNABert_tokenize.py --data_set Hartwig
python BulkRNABert_tokenize.py --data_set TARGET_AML

# BulkFormer: maps each of the 20,010 genes to its ESM2-based gene identity
# token and rotary expression embedding. Produces per-sample dicts.
python BulkFormer_tokenize.py --data_set TCGA
# ... (repeat for other datasets)

# scGPT: selects top-5000 HVGs, rank-orders genes by expression per sample,
# bins into 51 adaptive bins. Produces (gene_ids, values) token arrays.
python scGPT_tokenize.py --data_set TCGA
# ... (repeat for other datasets)

# STATE-SE: selects top-5000 HVGs, applies soft binning into 10 learnable bins.
# Produces token arrays with ESM2-based gene identity encoding.
python STATE_tokenize.py --data_set TCGA
# ... (repeat for other datasets)

# CancerFoundation: selects top-1199 HVGs, applies adaptive per-sample bins.
# Produces (gene_ids, expression_values) token arrays.
python CancerFoundation_tokenize.py --data_set TCGA
# ... (repeat for other datasets)

# COMPASS: maps each sample to its gene expression matrix aligned with COMPASS
# gene vocabulary; also maps cancer types to TCGA project IDs.
python COMPASS_preprocess.py --data_set TCGA
# ... (repeat for other datasets)
```

Tokenized outputs are saved to `data/tokens/<FM>/<DATASET>_tokens.pkl`. These paths are configured per FM under `fm_configs.<FM>.tokens_paths` in `configs/finetune.yaml`.

#### Step 2 — Fine-tune

Each shell script runs 5 splits in parallel (one per GPU) and skips already-completed runs:

```bash
# Usage: ./scripts/train_finetune_<FM>.sh <DATASET>

bash scripts/train_finetune_BulkRNABert.sh    TCGA
bash scripts/train_finetune_BulkFormer.sh     TCGA
bash scripts/train_finetune_scGPT.sh          TCGA
bash scripts/train_finetune_STATE.sh          TCGA
bash scripts/train_finetune_COMPASS.sh        TCGA
bash scripts/train_finetune_CancerFoundation.sh TCGA
```

Run a single split directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.train_finetune \
  --config_path configs/finetune.yaml \
  --dataset TCGA \
  --fm_name BulkRNABert \
  --loss_name CIndex \
  --seed 37 \
  --split_idx 0
```

#### Fine-tuning configuration details

All settings are defined in `configs/finetune.yaml` under `fm_configs.<FM>`. Key parameters:

| FM | Method | LoRA targets | r / α | lr | Batch | Grad accum |
|----|--------|-------------|-------|----|-------|------------|
| BulkRNABert | LoRA | `w_q`, `w_k`, `w_v` | 8 / 16 | 5e-4 | 32 | 4 |
| BulkFormer | LoRA | `to_q`, `to_k`, `to_v`, `to_out` | 8 / 16 | 1e-5† | 6 | 22 |
| scGPT | LoRA | `out_proj`, `linear1`, `linear2` | 8 / 16 | 1e-4 | 32 | 4 |
| STATE-SE | LoRA | `qkv_proj`, `out_proj`, `linear1`, `linear2` | 8 / 16 | 1e-4 | 8 | 16 |
| CancerFoundation | LoRA | `out_proj`, `linear1`, `linear2` | 8 / 16 | 1e-4 | 32 | 4 |
| COMPASS | PFT‡ | N/A (no LoRA) | — | 1e-4 | 32 | 4 |

† Lower lr due to Performer FAVOR+ kernel gradient instability; gradient checkpointing enabled.  
‡ PFT = Partial Fine-Tuning: the input encoder is frozen, only the concept bottleneck and survival head are trained.

The effective batch size is 128 for all FMs (batch × grad_accum). Max epochs: 100, early stopping patience: 20.

Outputs are saved to `model_outputs_finetune/`. Evaluate with:

```bash
# Usage: ./scripts/eval_finetune.sh <FM_NAME> <DATASET>
bash scripts/eval_finetune.sh BulkRNABert    TCGA
bash scripts/eval_finetune.sh BulkFormer     TCGA
bash scripts/eval_finetune.sh scGPT          TCGA
bash scripts/eval_finetune.sh CancerFoundation TCGA
bash scripts/eval_finetune.sh COMPASS        TCGA
bash scripts/eval_finetune.sh STATE          TCGA

# Logs are written to: logs/eval_finetune/<DATASET>/<FM_NAME>.log
```

### Auxiliary Experiments

#### 1. Binned HVG Ablation

**Motivation:** Foundation models tokenize gene expression into discrete bins before processing. This experiment tests whether that discretization step alone — independent of the transformer architecture — is responsible for any performance difference relative to continuous HVG baselines.

**Method:** Continuous log-normalized HVG expression values are replaced with ordinal bin indices before being fed to the downstream survival model (ResidualMLP + CIndex). Bin edges are always fit on training data only and applied to validation and test sets. The best-performing HVG count per dataset is used as the fixed representation size (TCGA: 10,000; Hartwig: 5,000; CPTAC & TARGET-AML: 1,250).

**Binning strategies** (implemented in `pytorch_modules/data/binned_data.py`):

| Strategy | Description | Analogous FM |
|----------|-------------|--------------|
| `linear` | Equal-width bins per gene, spanning each gene's [min, max] across training samples | — |
| `quantile` | Equal-frequency bins per gene (percentile-based) | — |
| `global_linear` | Equal-width bins shared across all genes, spanning the global [min, max] | BulkRNABert |
| `global_quantile` | Equal-frequency bins shared across all genes (flattened training matrix) | — |
| `per_sample_linear` | Equal-width bins computed per sample | — |
| `per_sample_quantile` | Equal-frequency bins computed per sample (per-sample rank-based) | scGPT, CancerFoundation |

**Bin counts swept:** {2, 8, 16, 32, 51, 64, 128, 256, 512, 1024}

Total runs: 10 bin counts × 6 strategies × 5 splits × 4 datasets = 1,200 models (distributed across 8 GPUs, 2 per dataset).

```bash
# Launch all datasets across 8 GPUs (strategies A on GPUs 0–3, strategies B on GPUs 4–7)
bash scripts/train_binned_HVG_models.sh

# Manual single-GPU mode:
# ./scripts/train_binned_HVG_models.sh <GPU_ID> <DATASET> <STRATEGY...>
bash scripts/train_binned_HVG_models.sh 0 TCGA linear global_linear per_sample_linear
bash scripts/train_binned_HVG_models.sh 4 TCGA quantile global_quantile per_sample_quantile
```

Or a single run directly:

```bash
python -m train.train_binned_HVG_model \
  --config_path configs/all_models.yaml \
  --dataset TCGA \
  --n_HVGs 10000 \
  --n_bins 64 \
  --bin_strategy global_linear \
  --arch ResidualMLP \
  --loss_name CIndex \
  --seed 37 \
  --split_idx 0 \
  --hp_path hp_results/all_aggregated_hps.json
```

Outputs go to `model_outputs/<DATASET>/survival/PFS/binned_HVG_ablation/`. Evaluate with:

```bash
bash scripts/run_eval_binned_HVG.sh
# or: python -m test.eval_binned_HVG_models --dataset TCGA
```

#### 2. Mean-Pooled FM Token Embeddings (HVG Pooling)

**Motivation:** Standard FM inference uses the CLS token (or an equivalent pooled representation) as the sample embedding. However, both BulkRNABert and BulkFormer also produce a contextualized embedding for *each individual gene token*. This experiment tests whether averaging the per-gene token embeddings over a subset of HVGs — rather than using the CLS token — can improve the prognostic signal extracted from these models.

**Method:** For BulkRNABert and BulkFormer, the model is run with `embeddings_layers_to_save` enabled to extract intermediate-layer per-gene embeddings. The gene-level embeddings for the top-k HVGs (selected by log-variance) are mean-pooled to produce a fixed-dimensional representation. This is implemented in `data_preprocessing/get_FM_embeddings/BulkRNABert_HVG_pooling.py` (and equivalently for BulkFormer).

**HVG counts swept:**
- BulkRNABert: {1,250; 5,000; 10,000}
- BulkFormer: {1,250; 2,000; 5,000; 10,000}

The embedding keys used in training are `BulkRNABert_HVG_<k>` and `BulkFormer_HVG_<k>`. This experiment was run on TCGA only, with both seeds and all 5 splits, using both architectures and loss functions (2 archs × 2 losses = 8 concurrent jobs per GPU).

**Step 1 — Extract HVG-pooled embeddings** (run from `data_preprocessing/get_FM_embeddings/`):

```bash
python BulkRNABert_HVG_pooling.py --data_set TCGA
```

**Step 2 — Assemble into the main `.pkl` file:**

```bash
python -m data_preprocessing.TCGA.augment_TCGA_with_HVG_embeddings
```

**Step 3 — Train survival models:**

```bash
# GPU 6: BulkRNABert HVG sizes {1250, 5000, 10000}
# GPU 7: BulkFormer  HVG sizes {1250, 2000, 5000, 10000}
bash scripts/train_TCGA_HVG_pooling.sh
```

Or a single run:

```bash
python -m train.train_RNA_model_on_pooled_HVG_embeds \
  --config_path configs/all_models_HVG_pooling.yaml \
  --dataset TCGA \
  --task survival \
  --arch ResidualMLP \
  --loss_name CIndex \
  --rna_embedding_key BulkRNABert_HVG_10000 \
  --seed 37 \
  --split_idx 0 \
  --hp_path hp_results/all_aggregated_hps.json
```

Outputs go to `model_outputs/HVG_pooling/TCGA/survival/PFS/`. Evaluate with:

```bash
python -m test.eval_all_RNA_pooled_HVG --dataset TCGA --endpoint PFS
```

#### 3. Linear Probing

**Motivation:** Assesses how much *biological structure* is captured in each embedding representation — independently of prognostic modeling — by evaluating the linear separability of clinically meaningful categorical labels.

**Method:** A bias-free linear classifier (`nn.Linear`, no bias) is trained on each embedding with cross-entropy loss and early stopping (patience = 50 epochs). The classifier is evaluated on three classification tasks, applied to the datasets where labels are available:

| Task | Datasets | Metric |
|------|----------|--------|
| `cancer_type` — predict cancer type | TCGA, CPTAC, Hartwig | Accuracy, Macro F1 |
| `organ` — predict primary tumor organ | TCGA, CPTAC, Hartwig | Accuracy, Macro F1 |
| `grade` — predict tumor grade (AJCC) | TCGA, CPTAC | Accuracy, Macro F1, Quadratic Weighted Kappa |

All 25 embedding specs from the main benchmarking are evaluated (6 FMs, 8 HVG sizes, 4 gene-set × pooling combos, 2 PCA configs, raw expression), across all 5 splits. Implemented in `mech/linear_probing.py`.

```bash
# Runs 4 datasets × 8 embedding-range workers in parallel across GPUs 4–7
bash scripts/run_linear_probing.sh
```

The shell script splits the 25 embedding specs into 8 non-overlapping index ranges and runs them in parallel per dataset, then merges the partial CSVs:

```bash
# Direct invocation for a single dataset (all embeddings):
python -m mech.linear_probing \
  --dataset TCGA \
  --device cuda:0 \
  --out_csv benchmarking_results/linear_probe/linear_probing_results.csv

# With embedding range (for parallelism, indices into EMBEDDING_SPECS list):
python -m mech.linear_probing \
  --dataset TCGA \
  --device cuda:0 \
  --emb_start 0 \
  --emb_end 6 \
  --out_csv benchmarking_results/linear_probe/tmp/partial_TCGA_0_6.csv
```

Supports `--overwrite` to rerun from scratch and incremental saving (results written after each dataset so no progress is lost on interruption). Results are written to `benchmarking_results/linear_probe/linear_probing_results.csv`.

---

## Evaluation

Each experiment type has its own evaluation script. Evaluation commands are co-located with their training instructions above. The table below summarizes all evaluation entry points:

| Experiment | Eval script | Shell launcher |
|------------|-------------|----------------|
| Frozen embeddings (main benchmarking) | `test.eval_all_RNA_models` | `scripts/run_eval_all_datasets.sh` |
| FM + HVG concatenation | `test.eval_all_RNA_models_concat` | `scripts/run_eval_all_datasets_concat.sh` |
| FM + Clinical concatenation | `test.eval_all_RNA_models_clinical` | `scripts/run_eval_all_datasets_clinical.sh` |
| Fine-tuned FMs | `test.eval_finetune` | `scripts/eval_finetune.sh <FM> <DATASET>` |
| Binned HVG ablation | `test.eval_binned_HVG_models` | `scripts/run_eval_binned_HVG.sh` |
| HVG-pooled FM embeddings | `test.eval_all_RNA_pooled_HVG` | — |
| Linear probing | `mech.linear_probing` | `scripts/run_linear_probing.sh` |

### Frozen Embeddings (Main Benchmarking)

```bash
# All 4 datasets in parallel:
bash scripts/run_eval_all_datasets.sh

# Single dataset:
CUDA_VISIBLE_DEVICES=0 python -m test.eval_all_RNA_models \
  --dataset TCGA --task survival --endpoint PFS
```

The evaluator scans `model_outputs/<DATASET>/survival/PFS/` for run folders, loads each checkpoint, runs inference on the test set, and computes the C-index. Results are written to `all_metrics.csv` in the results directory.

### Validation C-Index (for Model Selection)

Computes validation C-indices across all runs to select the best architecture/loss configuration per embedding:

```bash
bash scripts/run_compute_val_cindex.sh
# Runs all 4 datasets in parallel on GPUs 4–7; logs to logs/val_cindex/
```

---

## Analysis

The `notebooks/` directory contains Jupyter notebooks for analyzing results and generating paper figures:

- **`analyze_results.ipynb`** — Main results analysis: loads the aggregated CSVs, computes mean C-index ± SD per embedding/dataset, generates the paper's main tables and heatmaps (Figures 3–5), and runs the sign-flip permutation test for statistical significance.
- **`analyze_hp_results.ipynb`** — Analyzes hyperparameter tuning convergence and identifies best HPs across configurations.
- **`analyze_datasets.ipynb`** — Dataset statistics: patient counts, gene count distributions, PFS event rates, cancer type breakdowns.
- **`analyze_clinical_metadata.ipynb`** — Explores clinical feature availability and distributions across cohorts.

### Extracting Linear Model Gene Weights

The `analysis/extract_linear_weights.py` script loads trained Linear CIndex survival models and extracts the learned gene weights (mapping each gene to its contribution to the risk score), enabling identification of the most prognostically informative genes:

```bash
python analysis/extract_linear_weights.py
# Output: analysis/linear_gene_weights.pkl
```

---

## Model Architecture & Loss Functions

### Architectures

Two downstream predictor architectures were evaluated (implemented in `pytorch_modules/nn/`):

**ResidualMLP** (`pytorch_modules/nn/residual_MLP.py`):
- Input projection: Linear(d_in → 512) + GELU + Dropout
- 3 residual blocks: each with two FC layers (512 → 512), GELU activation, skip connection, and ReLU + Dropout
- Output: Linear(512 → 1, bias=False)

**Linear** (`pytorch_modules/nn/MLP.py`):
- Single linear layer: Linear(d_in → 1, bias=False)

Both are wrapped in `pytorch_modules/models/survival_models.py` as `RNA_SurvivalModel` (for survival tasks) and `RNA_ResponseModel` (for treatment response classification), implemented as PyTorch Lightning modules.

### Loss Functions

Two loss functions are implemented in `pytorch_modules/losses/`:

**Pairwise Ranking Loss (CIndex)** — `cindex_loss.py`:
Optimizes the C-index directly via a logistic approximation over comparable patient pairs (i, j) where patient i progresses before j:
$$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{C}|} \sum_{(i,j) \in \mathcal{C}} \log(1 + e^{-(r_i - r_j)})$$

**Cox Partial Log-Likelihood (Cox)** — `cox_loss.py`:
Minimizes the negative Cox partial log-likelihood with Breslow tie handling and minibatch risk-set approximation (batch size 128 → effective batch size 512 with gradient accumulation):
$$\mathcal{L}_{\text{Cox}} = -\frac{1}{\sum_i \delta_i} \sum_{i:\delta_i=1} \left( r_i - \log \sum_{j \in \mathcal{R}(t_i)} e^{r_j} \right)$$

An elastic net penalty is applied on top of either loss via `elastic_loss.py`.

### Training Details

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Effective batch size | 512 (4× gradient accumulation) |
| Max epochs | 500 |
| Early stopping patience | 50 epochs |
| Gradient clipping | 1.5 |
| Optimizer | AdamW |
| Hardware | NVIDIA RTX A6000 (48 GB) |
| Seeds | 37 and 42 |
| Splits | 5 per dataset |

---

## Configuration

### `configs/all_models.yaml`

The main configuration file for frozen-embedding experiments. Specifies:
- Dataset-specific `.pkl` data paths (`data_paths`)
- Dataset-specific split folders (`split_folders`)
- Shared data settings (batch size, num workers, PFS endpoint)
- Default model hyperparameters per architecture (lr, weight decay, hidden dim, dropout)
- Trainer settings (max epochs, gradient clipping, gradient accumulation)
- Callback settings (early stopping patience, monitor metric)

### `configs/finetune.yaml`

Configuration for fine-tuning experiments. Extends `all_models.yaml` with:
- FM-specific LoRA settings (`lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`)
- FM-specific token file paths (`tokens_paths`)
- FM-specific batch size and gradient accumulation overrides
- Fine-tuning mode for COMPASS (`ft_mode: PFT`)

### `configs/all_models_HVG_pooling.yaml`

Configuration for the HVG-pooled FM embedding ablation experiment.

---

## Outputs & Results

| Directory | Contents |
|-----------|----------|
| `model_outputs/<DATASET>/survival/PFS/` | Frozen-embedding model checkpoints + `all_metrics.csv` |
| `model_outputs/<DATASET>/survival/PFS/binned_HVG_ablation/` | Binned HVG ablation model checkpoints |
| `model_outputs/HVG_pooling/<DATASET>/survival/PFS/` | Mean-pooled FM token embedding model checkpoints |
| `model_outputs_finetune/<DATASET>/survival/PFS/` | Fine-tuned model checkpoints |
| `concat_model_outputs/<DATASET>/survival/PFS/` | FM + HVG concatenation model checkpoints |
| `hp_results/` | Per-split HP tuning JSONs + `all_aggregated_hps.json` |
| `splits/<DATASET>/` | Train/val/test split `.pkl` files |
| `benchmarking_results/` | Final aggregated evaluation CSVs |
| `logs/` | Training and evaluation log files |

Each run folder under `model_outputs/` is named:
```
{arch}_{loss}_rnaKey_{embedding_key}_nHVGs_{n}_seed_{seed}_split_{split_idx}_{timestamp}/
```
and contains a model checkpoint (`.ckpt`) and a run configuration file (`config.json`).

---

## Contact

- Ananth Shyamal — ashyamal@mit.edu
- Ling Min Serena Khoo — klingmin@mit.edu
- Regina Barzilay — regina@csail.mit.edu

# Do Transcriptomic Foundation Models Improve Clinical Prediction?

Code for the paper **"Do Transcriptomic Foundation Models Improve Clinical Prediction?"** by Ananth Shyamal, Ling Min Serena Khoo, and Regina Barzilay (MIT CSAIL; University of Michigan Medical School).

We benchmark seven bulk and single-cell transcriptomic foundation models (FMs) against four simple transcriptomic baselines on two clinical prediction tasks from bulk tumor RNA-seq:

1. **Progression-free survival (PFS)** across four cancer cohorts (TCGA, Hartwig, CPTAC, TARGET-AML; 15,665 patients).
2. **Immune checkpoint inhibitor (ICI) response** across five immunotherapy cohorts (628 patients) under three generalization paradigms: within-cohort, leave-one-cohort-out (LOCO), and cross-cohort transfer.

**Main findings**
- Every FM underperformed all four baselines for PFS prediction on TCGA (best FM: STATE-SE, C-index 0.728; best baseline: 5,000 HVGs, 0.742). The same trend held on Hartwig, CPTAC, and TARGET-AML, and in 27 of 33 TCGA cancer types.
- For ICI response, most FMs underperformed the best baseline in all three paradigms. COMPASS, which was designed for ICI response prediction, only matched the gene-set baseline.
- Linear probes and UMAPs show that FM embeddings capture primary tumor site and tumor grade less well than simple baselines.
- Discretizing expression into bins, a step used by several FMs, consistently reduced HVG performance for PFS prediction.
- LoRA fine-tuning of scGPT and CancerFoundation narrowed but did not close the gap to baselines.
- The best transcriptomic models outperformed Cox models based on clinical covariates (primary tumor site, stage, demographics), so bulk RNA-seq does carry prognostic information that current FMs do not fully capture.

---

## Contents

- [What is benchmarked](#what-is-benchmarked)
- [Paper figures and tables → code](#paper-figures-and-tables--code)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Data](#data)
- [PFS benchmark](#pfs-benchmark)
- [ICI response benchmark](#ici-response-benchmark)
- [Additional analyses](#additional-analyses)
- [Model details](#model-details)
- [Outputs](#outputs)
- [Other experiments in this repository](#other-experiments-in-this-repository)
- [Contact](#contact)

---

## What is benchmarked

**Foundation models (7)**, used as frozen feature extractors:

| Model | Type | Input genes | Gene selection | Embedding dim |
|---|---|---|---|---|
| BulkRNABert | Bulk | 19,062 | All shared genes | 256 |
| BulkFormer | Bulk | 20,010 | All shared genes | 640 |
| COMPASS | Bulk | 15,672 | All shared genes (+ cancer type) | 44 |
| scGPT | Single-cell | 5,000 | Top HVGs | 512 |
| STATE-SE | Single-cell | 5,000 | Top HVGs | 2,058 |
| CancerFoundation | Single-cell | 1,199 | Top HVGs | 256 |
| UCE | Single-cell | 1,024 | Expression-weighted sampling | 1,280 |

Single-cell models treat each bulk sample as a pseudo single cell.

**Transcriptomic baselines (25 variants in 4 families):**
- **HVGs:** top-*k* genes by log-scale variance, *k* ∈ {18, 44, 50, 132, 256, 500, 512, 640, 729, 1,250, 1,280, 2,058, 5,000, 10,000}. The sizes 44, 256, 512, 640, 1,280 and 2,058 match FM embedding dimensions (Figure S1). For PFS, HVGs are selected on each split's training set; for ICI, the TCGA variance ranking is used.
- **Gene sets:** max pooling and ssGSEA over Hallmark (50 sets), Immuno (132), AURORA (729), and Combined (910).
- **PCA:** 50 and 100 components, fit on each training split.
- **Raw expression:** all protein-coding genes passing the expression filter.

**Downstream predictors:** a linear layer and a residual MLP, each trained with a Cox partial-likelihood loss and a pairwise ranking (C-index) loss for PFS, and with class-weighted binary cross-entropy for ICI response.

In total, 32 embeddings × 2 architectures × 2 losses × 2 seeds × 5 splits × 4 cohorts = 5,120 PFS models, plus the ICI models for each paradigm.

---

## Paper figures and tables → code

Figures are written to `paper_figures/` by the two analysis notebooks.

| Paper item | Content | Produced by | Output |
|---|---|---|---|
| Fig. 2A | TCGA PFS C-index per FM / baseline (with permutation-test stars) | `notebooks/analyze_PFS_benchmarking_results.ipynb` | `tcga_results_bar.png` |
| Fig. 2B | Per-cancer-type ΔC-index on TCGA | PFS notebook | `TCGA_delta_cindex_dotplot.png` |
| Fig. 2C | Best bulk FM / single-cell FM / baseline across 4 cohorts | PFS notebook | `all_datasets_results_radar.png` |
| Fig. 2D | ICI AUPRC per category, 3 paradigms | `notebooks/analyze_ICI_benchmarking_results.ipynb` | `ICI_auprc_overview.png` |
| Fig. S1 | FM vs. dimension-matched HVGs | PFS notebook | `fm_vs_matched_dim_baselines.png` |
| Fig. S2 | Frozen vs. LoRA fine-tuned scGPT / CancerFoundation | PFS notebook | `finetune_comparison.png` |
| Fig. S3 | Linear probing (primary tumor site, tumor grade) | `mech/linear_probing.py` → PFS notebook | `linear_probing_tcga_bars.png`, `linear_probing_heatmap_site_grade.png` |
| Fig. S4 | TCGA UMAPs colored by site and grade | PFS notebook | `TCGA_UMAP_clinical_features.png` |
| Fig. S5 | HVG binning ablation | `train/train_binned_HVG_model.py` → PFS notebook | `binned_HVG_ablation*.png` |
| Figs. S6, S7 | Per-cancer-type heatmaps (TCGA, CPTAC) | PFS notebook | `TCGA_per_cancer_type_heatmap.png`, `CPTAC_per_cancer_type_heatmap.png` |
| Fig. S8 | C-index vs. number of HVGs | PFS notebook | `hvg_line_chart.png` |
| Table S6 | Hyperparameter search spaces and tuned embeddings | `train/tune.py`, `train/tune_ICI.py` | `hp_results/all_aggregated_hps.json`, `hp_results_ICI/all_aggregated_hps_ICI.json` |
| Tables S8, S9 (clinical rows) | Clinical covariate Cox baselines | `train/train_simple_clinical_baseline.py` | `benchmarking_results/clinical_baseline_results*.csv` |
| Table S9 | PFS C-index per category, 4 cohorts | PFS notebook | `benchmarking_results/transcriptomic/*_survival_PFS_agg_filtered.csv` |
| Table S10 | ICI AUPRC per category, 3 paradigms | ICI notebook | printed summary; `benchmarking_results/ICI/per_task_best_auprc.csv` |
| Table S11 | Cancer types where FMs beat baselines | PFS notebook | `benchmarking_results/transcriptomic/fm_vs_baseline_cancer_type_summary.csv` |
| Table S12 | Binned HVGs vs. FMs with native binning | PFS notebook | — |
| Sign-flip permutation tests (Note S1E) | FM vs. best baseline, per dataset | PFS notebook | `benchmarking_results/stats/permutation_test_results.csv` |

---

## Repository structure

```
RNA-FM-benchmarking/
├── configs/
│   ├── all_models.yaml                 # Main PFS config (data paths, splits, trainer, defaults)
│   ├── immunotherapy_cohorts.yaml      # ICI cohort registry (paths, cancer type, COMPASS codes)
│   ├── finetune.yaml                   # LoRA / fine-tuning config
│   └── all_models_HVG_pooling.yaml     # Config for the HVG-pooled FM experiment
│
├── data_preprocessing/
│   ├── TCGA/ Hartwig/ CPTAC/ TARGET_AML/   # Per-cohort preprocessing notebooks + embedding assembly
│   ├── immunotherapy_cohorts/          # ICI cohort processing (GEO + EGA) and embedding assembly
│   ├── get_FM_embeddings/              # FM embedding extraction (BulkRNABert, BulkFormer, COMPASS,
│   │                                   #   scGPT, STATE, UCE, CancerFoundation)
│   ├── compute_PCA.py                  # PCA projections
│   ├── get_ssGSEA_scores.py            # ssGSEA gene-set scores
│   ├── get_genes_sets.py, dict_to_gmt.py
│   ├── get_data_splits.py              # PFS splits (5 × 70/15/15, patient-level)
│   ├── get_ICI_data_splits.py          # ICI splits for within-cohort / LOCO / transfer
│   └── tokenize_for_finetuning/        # Pre-tokenization for FM fine-tuning
│
├── pytorch_modules/
│   ├── data/          # Lightning DataModules: data.py (PFS), ici_data.py (ICI),
│   │                  #   binned_data.py (binning ablation), finetune_data.py (fine-tuning)
│   ├── models/        # survival_models.py (PFS / response heads), finetune_models.py (LoRA FMs)
│   ├── nn/            # MLP.py (linear), residual_MLP.py
│   ├── losses/        # cox_loss.py, cindex_loss.py, elastic_loss.py
│   └── utils/
│
├── train/             # Training entry points (PFS, ICI, tuning, clinical baseline, ablations)
├── val/               # compute_val_cindex.py: validation C-index for model selection
├── test/              # Evaluation entry points (PFS, ICI, COMPASS ICI, binning, fine-tuning)
├── mech/              # linear_probing.py
├── scripts/           # Shell launchers (multi-GPU sweeps); scripts/embeddings/ for FM extraction
├── notebooks/         # analyze_PFS_benchmarking_results.ipynb, analyze_ICI_benchmarking_results.ipynb, …
├── splits/            # Split files used in the paper (TCGA_cleaned, Hartwig, CPTAC, TARGET_AML, ICI)
├── hp_results/        # PFS tuning results + all_aggregated_hps.json
├── hp_results_ICI/    # ICI tuning results + all_aggregated_hps_ICI.json
├── benchmarking_results/  # Aggregated metrics tables used for the paper
└── paper_figures/     # Figures generated by the notebooks
```

Not included in the repository: `data/` (raw and processed data), `model_outputs*/` (checkpoints), `logs/`, and `wandb/`.

---

## Setup

The core benchmarking environment:

```bash
conda env create -f env.yml        # or: conda create -n rna_modeling python=3.12 && pip install -r requirements.txt
conda activate rna_modeling
```

Each FM's embedding extraction runs in its own environment because of dependency conflicts. The launchers in `scripts/embeddings/` activate the right one:

| Model | Environment |
|---|---|
| BulkRNABert | `rna_modeling` |
| BulkFormer | `bulkformer` |
| COMPASS | `compass` |
| scGPT | `scgpt_env` |
| STATE-SE | `state` |
| CancerFoundation | `cancerfoundation` |
| UCE | `uce` |

Model weights and code for each FM come from the authors' releases. Several scripts contain absolute paths to our cluster (for example `/data/rbg/users/ashyamal/...`) for FM checkpoints and data; update these to your own locations.

Run all Python entry points from the **repository root** as modules, for example `python -m train.train_RNA_model`.

---

## Data

### PFS cohorts

| Cohort | Patients | Samples | Setting | Source |
|---|---|---|---|---|
| TCGA | 10,388 | 10,388 | Primary pan-cancer (33 types) | NCI GDC; PFS from the TCGA Pan-Cancer Clinical Data Resource |
| Hartwig | 2,153 | 2,232* | Metastatic pan-cancer | Hartwig Medical Foundation (data access request) |
| CPTAC | 1,222 | 1,222 | Primary pan-cancer | NCI GDC |
| TARGET-AML | 1,902 | 2,586 | Pediatric AML | Xena Browser (TPM) + GDC (clinical) |

\*8 Hartwig samples were removed for data-access reasons, leaving 2,224 for model development.

### ICI cohorts

| Cohort | n | Responders | Cancer type | Source |
|---|---|---|---|---|
| IMvigor210 | 298 | 68 | Bladder urothelial | EGA EGAD00001003977 |
| IMmotion150 | 165 | 48 | Renal cell | EGA EGAD00001004183 (atezolizumab arms only) |
| Rose | 89 | 16 | Bladder urothelial | GEO GSE176307 |
| Riaz | 51 | 10 | Melanoma | GEO GSE91061 |
| Hugo | 25 | 13 | Melanoma | GEO GSE78220 |

Response labels (CR/PR vs. SD/PD) for Hugo, Riaz, and Rose come from the [COMPASS dashboard](https://www.immuno-compass.com/download/index.html). IMvigor210 and IMmotion150 were aligned with STAR against GENCODE v36 and converted to TPM, reproducing the GDC mRNA pipeline (Note S1, A.2).

### Common preprocessing

All data are TPM. Genes with mean TPM < 0.1 are removed. Baselines use protein-coding genes only. For the PFS cohorts, FMs receive all genes passing the filter and apply their own gene selection; for the ICI cohorts, all methods use protein-coding genes. ICI expression is aligned to the TCGA gene vocabulary (17,927 genes) so that features are defined consistently across cohorts.

Download links and full details are in Note S1 of the Supplementary Information.

---

## PFS benchmark

### 1. Preprocess each cohort

Run the notebook in each `data_preprocessing/<COHORT>/` folder (`process_tcga.ipynb`, `preprocess_Hartwig.ipynb`, `preprocess_CPTAC.ipynb`, `preprocess_TARGET_AML.ipynb`). `TCGA/process_cleaned_TCGA_clinical.ipynb` adds the curated Pan-Cancer PFS labels.

### 2. Extract FM embeddings

```bash
bash scripts/embeddings/run_all_embeddings.sh --dataset TCGA --gpu 0      # all FMs except BulkRNABert, + ssGSEA and PCA

# or individually:
bash scripts/embeddings/run_BulkRNABert.sh      --dataset TCGA --gpu 0
bash scripts/embeddings/run_BulkFormer.sh       --dataset TCGA --gpu 0
bash scripts/embeddings/run_COMPASS.sh          --dataset TCGA --gpu 0
bash scripts/embeddings/run_scGPT.sh            --dataset TCGA --gpu 0
bash scripts/embeddings/run_STATE.sh            --dataset TCGA
bash scripts/embeddings/run_CancerFoundation.sh --dataset TCGA --gpu 0
bash scripts/embeddings/run_UCE.sh              --dataset TCGA --gpu 0
```

### 3. Compute baselines and assemble per-cohort data

```bash
bash scripts/embeddings/run_ssGSEA.sh --dataset TCGA       # gene-set scores

python -m data_preprocessing.TCGA.get_TCGA_embeddings       # one .pkl per cohort with all embeddings
python -m data_preprocessing.Hartwig.get_Hartwig_embeddings --data_key Hartwig
python -m data_preprocessing.CPTAC.get_CPTAC_embeddings
python -m data_preprocessing.TARGET_AML.get_TARGET_AML_embeddings
```

The assembled files go to `data/RNA_embeddings/<COHORT>_processed*.pkl`. HVG selection and PCA are fit on each split's training set at training time.

### 4. Splits

```bash
python -m data_preprocessing.get_data_splits --data_set TCGA    # also CPTAC, Hartwig, TARGET_AML
```

This creates five patient-level 70/15/15 splits per cohort, stratified by cancer type (TCGA), organ (CPTAC), or cancer type × response (Hartwig). The exact splits used in the paper are in `splits/`.

### 5. Hyperparameter tuning (TCGA)

We tuned 12 representative embeddings with Optuna TPE: the 7 FMs, 1,250 HVGs (shared by all HVG sizes), combined gene sets with max and ssGSEA pooling (shared within each pooling strategy), PCA-50 (shared with PCA-100), and raw expression. Each was tuned for every architecture × loss, with 50 trials per split on 5 splits.

```bash
# ./scripts/tune_RNA_models.sh <RNA_EMBEDDING_KEY> <ARCH> <LOSS_NAME>
bash scripts/tune_RNA_models.sh BulkFormer   ResidualMLP CIndex
bash scripts/tune_RNA_models.sh RNA_seq_1250 Linear      Cox

python -m train.agg_best_hps     # → hp_results/all_aggregated_hps.json
```

The final hyperparameters are the geometric mean (learning rate, weight decay, penalty weight) or arithmetic mean (dropout) of the five per-split optima.

| Hyperparameter | Range | Distribution |
|---|---|---|
| Learning rate | [1e-5, 1e-3] | Log-uniform |
| Weight decay | [1e-7, 1e-4] | Log-uniform |
| Penalty weight (elastic net) | [1e-4, 1e-2] | Log-uniform |
| Dropout | [0.1, 0.5] | Uniform (residual MLP only) |

### 6. Train

```bash
# ./scripts/train_RNA_models.sh <GPU_ID> <DATASET> survival <EXPERIMENT> <ARCH> <LOSS_NAME>
#   EXPERIMENT: HVG | HVG1 | HVG2 | HVG3 | HVG4 (FM-matched sizes) | GS | SSGSEA | MAX |
#               FM_BULK | FM_SC | PCA | BASELINE (raw expression) | ALL
SEED=37 bash scripts/train_RNA_models.sh 0 TCGA survival FM_SC ResidualMLP CIndex
SEED=42 bash scripts/train_RNA_models.sh 1 TCGA survival HVG   Linear      Cox
```

Already-completed runs are skipped. To run everything for one cohort across GPUs:

```bash
bash scripts/run_all_single_dataset.sh TCGA
```

A single run:

```bash
python -m train.train_RNA_model \
  --config_path configs/all_models.yaml --dataset TCGA --task survival \
  --arch ResidualMLP --loss_name CIndex --rna_embedding_key RNA_seq --n_HVGs 5000 \
  --seed 37 --split_idx 0 --hp_path hp_results/all_aggregated_hps.json
```

### 7. Evaluate

```bash
bash scripts/run_compute_val_cindex.sh     # validation C-index → model_outputs/<DATASET>/survival/PFS/val_metrics.csv
bash scripts/run_eval_all_datasets.sh      # test C-index → model_outputs/<DATASET>/survival/PFS/all_metrics.csv
```

Architecture and loss are chosen per embedding on mean validation C-index, as is the best variant within each baseline family. We report mean test C-index over 5 splits × 2 seeds.

Both scoring scripts (`test/eval_all_RNA_models.py`, `val/compute_val_cindex.py`) can restrict scoring to a common set of patients, so every representation is compared on the same people. The models are not retrained:

```bash
# Only patients that COMPASS runs use (CPTAC and Hartwig samples without a TCGA cancer-type mapping are dropped)
python -m test.eval_all_RNA_models --dataset CPTAC --task survival --compass_subset   # -> all_metrics_compass_subset.csv
python -m val.compute_val_cindex   --dataset CPTAC --compass_subset                   # -> val_metrics_compass_subset.csv

# TARGET-AML: one sample per person, and no val/test person who also appears in train
# (the same person can appear as TARGET-20-X and TARGET-21-X)
python -m data_preprocessing.TARGET_AML.make_clean_eval_ids                           # -> splits/TARGET_AML/clean_eval_ids.json
python -m test.eval_all_RNA_models --dataset TARGET_AML --task survival --eval_ids_file splits/TARGET_AML/clean_eval_ids.json   # -> all_metrics_clean.csv
python -m val.compute_val_cindex   --dataset TARGET_AML --eval_ids_file splits/TARGET_AML/clean_eval_ids.json                   # -> val_metrics_clean.csv
```

### 8. Analyze

Run `notebooks/analyze_PFS_benchmarking_results.ipynb` from top to bottom. It aggregates metrics, runs the sign-flip permutation tests, and writes the PFS figures and tables.

---

## ICI response benchmark

### 1. Preprocess cohorts

- Hugo, Riaz, Rose (GEO): `data_preprocessing/immunotherapy_cohorts/process_ICI_cohorts.ipynb`
- IMvigor210, IMmotion150 (EGA, STAR/GENCODE v36 TPM):
  ```bash
  python data_preprocessing/immunotherapy_cohorts/process_ega_cohorts.py
  ```

Cohort metadata (cancer type, COMPASS cancer code, drug) is in `configs/immunotherapy_cohorts.yaml`.

### 2. Embeddings, assembly, and splits

```bash
bash scripts/embeddings/run_all_ICI_embeddings.sh      # FMs + ssGSEA + PCA for all 5 cohorts
# or per cohort: bash scripts/embeddings/run_ICI_embeddings.sh --cohort Hugo --gpu 0

for C in Hugo Riaz Rose IMMotion150 IMVigor210; do
  python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort $C
done
python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge   # → data/RNA_embeddings/ICI/ICI_all_processed.pkl

python data_preprocessing/get_ICI_data_splits.py --paradigm within_cohort
python data_preprocessing/get_ICI_data_splits.py --paradigm loco
python data_preprocessing/get_ICI_data_splits.py --paradigm transfer
```

**Paradigms**
- **Within-cohort:** 5 stratified 70/15/15 splits per cohort. The paper reports IMvigor210 and IMmotion150.
- **LOCO:** each cohort held out in turn; the other four are split 85/15 into train and validation.
- **Transfer:** train on one source cohort (85/15 train/val), test on a different target cohort. The paper reports the 8 pairs with IMvigor210 or IMmotion150 as the source.

### 3. Tune, train, evaluate

```bash
bash scripts/tune_ICI_models.sh             # tuned on IMvigor210 within-cohort splits
python -m train.agg_best_hps_ICI            # → hp_results_ICI/all_aggregated_hps_ICI.json

# ./scripts/train_ICI_models.sh <GPU_ID> <PARADIGM> <EXPERIMENT> <ARCH>
bash scripts/train_ICI_models.sh 0 within_cohort FM_SC ResidualMLP
bash scripts/run_all_ICI_benchmarking.sh    # full 8-GPU sweep over all paradigms

bash scripts/run_eval_ICI.sh 0              # → model_outputs/ICI/<paradigm>/<cohort_tag>/{all,val}_metrics.csv
```

The primary metric is AUPRC; AUROC, MCC, and macro F1 are also computed. For each paradigm, one architecture per embedding, and one variant per baseline family, is selected on validation AUPRC averaged over that paradigm's tasks.

### 4. COMPASS fine-tuning

Because COMPASS was built for ICI response, we also evaluate it with task-specific fine-tuning:

```bash
bash scripts/train_compass_ICI.sh [--paradigm loco|within_cohort|transfer] [--mode LFT|PFT|FFT]
bash scripts/eval_compass_ICI.sh  [--paradigm ...]
```

### 5. Analyze

Run `notebooks/analyze_ICI_benchmarking_results.ipynb` (Figure 2D, Table S10, per-cohort and COMPASS comparisons).

---

## Additional analyses

### Clinical covariate baselines (Tables S8, S9)

Cox proportional hazards models (`CoxPHSurvivalAnalysis`, L2 α = 0.1) fit on routinely collected clinical fields, using the same splits and one sample per patient:

```bash
python train/train_simple_clinical_baseline.py
# → benchmarking_results/clinical_baseline_results.csv (per split) and clinical_baseline_results_wide.csv (table)
```

Covariate sets: demographics and stage; primary tumor site only; and both combined. For the GDC cohorts, stage and grade are taken from the primary-disease diagnosis record.

### Linear probing (Figure S3) and UMAPs (Figure S4)

Bias-free linear probes (cross-entropy, Adam, early stopping) are trained on every embedding to predict **primary tumor site** (macro F1) and **two-tier tumor grade** (low G1–G2 vs. high G3–G4; Cohen's κ). The paper reports TCGA; the script also runs CPTAC and Hartwig where labels exist.

```bash
bash scripts/run_linear_probing.sh          # → benchmarking_results/linear_probe/linear_probing_results.csv
python -m mech.linear_probing --dataset TCGA --device cuda:0 --out_csv <path>   # single dataset
```

The UMAPs (TCGA; BulkFormer, BulkRNABert, STATE-SE, and 5,000 HVGs) are generated in the PFS notebook.

### Dimension-matched HVGs (Figure S1)

HVG baselines with *k* equal to each FM's embedding size (`EXPERIMENT=HVG4` in `train_RNA_models.sh`) test whether the FM gap is explained by dimensionality.

### Expression binning ablation (Figure S5, Table S12)

5,000 HVGs (TCGA) are discretized with six strategies (per-gene, global, or per-sample × linear or quantile) and 10 bin counts (2–1,024), mirroring the binning used by BulkRNABert (global linear, 64 bins) and by scGPT and CancerFoundation (per-sample quantile, 51 bins). Bin edges are fit on the training split only.

```bash
bash scripts/train_binned_HVG_models.sh
bash scripts/run_eval_binned_HVG.sh
```

Implementation: `pytorch_modules/data/binned_data.py`, `train/train_binned_HVG_model.py`, `test/eval_binned_HVG_models.py`.

### LoRA fine-tuning (Figure S2, Table S13)

scGPT and CancerFoundation fine-tuned end-to-end on TCGA PFS with LoRA (r = 8, α = 16, dropout 0.05 on `out_proj`, `linear1`, `linear2`), a linear survival head, C-index loss, and an effective batch size of 128.

```bash
cd data_preprocessing/tokenize_for_finetuning
python scGPT_tokenize.py --data_set TCGA
python CancerFoundation_tokenize.py --data_set TCGA
cd ../..

bash scripts/train_finetune_scGPT.sh TCGA
bash scripts/train_finetune_CancerFoundation.sh TCGA
bash scripts/eval_finetune.sh scGPT TCGA
bash scripts/eval_finetune.sh CancerFoundation TCGA
```

Settings are in `configs/finetune.yaml`.

---

## Model details

**Linear:** `Linear(d → 1)` (no bias for survival; bias for ICI classification).

**Residual MLP:** `Linear(d → 512)` + GELU + dropout, followed by 3 residual blocks (two 512-dimensional FC layers with GELU and dropout, a skip connection, then ReLU and dropout) and `Linear(512 → 1, bias=False)`.

**Losses** (`pytorch_modules/losses/`)
- *Pairwise ranking (C-index):* logistic approximation over comparable pairs, $\frac{1}{|\mathcal{C}|}\sum_{(i,j)\in\mathcal{C}}\log(1+e^{-(r_i-r_j)})$.
- *Cox:* negative partial log-likelihood, with risk sets approximated within each minibatch.
- *ICI:* binary cross-entropy with a positive-class weight of 3.05 (pooled non-responder/responder ratio).
- An elastic-net penalty (L1 + squared L2 on linear-layer weights), scaled by the tuned penalty weight, is added to every loss.

**Training:** batch size 128, 4 gradient-accumulation steps (effective batch size 512), AdamW, up to 500 epochs, early stopping patience 50, gradient clipping 1.5, seeds 37 and 42. Weights & Biases logging is optional (`USE_WANDB=1` in `scripts/train_RNA_models.sh`).

**Statistics:** one-sided sign-flip permutation test (FM vs. best baseline) on per-split C-indices averaged over seeds. With 5 splits, the minimum possible p-value is 1/32 ≈ 0.031.

---

## Outputs

| Location | Contents |
|---|---|
| `model_outputs/<DATASET>/survival/PFS/` | PFS run folders + `all_metrics.csv` (test) and `val_metrics.csv` |
| `model_outputs/ICI/<paradigm>/<cohort_tag>/` | ICI run folders + `all_metrics.csv` and `val_metrics.csv` |
| `model_outputs/ICI/compass/` | COMPASS ICI fine-tuning runs |
| `model_outputs_finetune/` | LoRA fine-tuning runs |
| `benchmarking_results/transcriptomic/` | Aggregated PFS tables per cohort |
| `benchmarking_results/ICI/` | Per-task ICI results |
| `benchmarking_results/stats/` | Permutation test results |
| `benchmarking_results/linear_probe/` | Linear probing results |
| `hp_results/`, `hp_results_ICI/` | Tuning histories and aggregated hyperparameters |

Run folders are named `{arch}_{loss}_rnaKey_{embedding}_nHVGs_{n}_seed_{seed}_split_{i}_{timestamp}/` and contain the checkpoint (`.ckpt`) and `config.json`.

---

## Other experiments in this repository

The following experiments are implemented but **not reported in the paper**:

- **FM + HVG concatenation:** `train/train_RNA_model_concat.py`, `scripts/train_RNA_models_concat.sh`, `test/eval_all_RNA_models_concat.py`
- **FM + clinical-feature concatenation:** `train/train_RNA_model_clinical.py`, `scripts/train_RNA_models_clinical.sh`, `test/eval_all_RNA_models_clinical.py`
- **Mean-pooled FM gene-token embeddings over HVGs:** `data_preprocessing/get_FM_embeddings/BulkRNABert_HVG_pooling.py`, `train/train_RNA_model_on_pooled_HVG_embeds.py`, `configs/all_models_HVG_pooling.yaml`
- **Fine-tuning other FMs** (BulkRNABert, BulkFormer, STATE-SE, COMPASS): `scripts/train_finetune_<FM>.sh`
- **Linear-model gene weights:** `analysis/extract_linear_weights.py`

---

## Contact

Corresponding authors: Ling Min Serena Khoo (klingmin@mit.edu) and Regina Barzilay (regina@csail.mit.edu).
Code questions: Ananth Shyamal (ashyamal@mit.edu).

"""
Data module for fine-tuning transcriptomic foundation models.

Loads pre-tokenized token IDs (saved by tokenize_for_finetuning/*.py) and
survival labels from the main processed data pickle.  Keeps the two concerns
separate so the tokenization only runs once and the main data file is unchanged.

Token formats
-------------
BulkRNABert      : tokens[pid] is np.int8 array of shape (n_genes,)
scGPT            : tokens[pid] is a dict {"gene_ids": np.int64, "expressions": np.float32}
                   Both arrays are variable-length (no padding).  Use scgpt_collate_fn.
CancerFoundation : same dict format as scGPT but gene_ids are shared across all patients
                   (fixed HVG set).  Use cancerfoundation_collate_fn (no scgpt import).
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_modules.utils.utils import load_pickle


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FinetuneTokensSurvivalDataset(Dataset):
    """Dataset that pairs pre-tokenized tokens with survival labels.

    Handles two token formats automatically:
      - np.ndarray (BulkRNABert): stored as ``input_ids`` LongTensor.
      - dict with ``gene_ids`` / ``expressions`` keys (scGPT): both stored
        as tensors under those names.

    Parameters
    ----------
    data : dict
        Mapping patient_id -> record dict (must contain endpoint_time,
        endpoint_event, and optionally organ, cancer_type).
    tokens : dict
        Mapping patient_id -> token representation (array or dict).
    endpoint : str
        Survival endpoint key prefix (e.g. "PFS").
    return_metadata : bool
        Whether to include organ / cancer_type in each batch item.
    metadata_keys : list[str]
        Metadata fields to return from the data dict.
    """

    def __init__(
        self,
        data: Dict[str, Dict],
        tokens: Dict[str, Any],
        endpoint: str,
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys or ["organ", "cancer_type"]

        common_ids = [pid for pid in data if pid in tokens]
        self.ids = common_ids
        self.data = {pid: data[pid] for pid in common_ids}
        self.tokens = {pid: tokens[pid] for pid in common_ids}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.ids[idx]
        rec = self.data[pid]
        tok = self.tokens[pid]

        t = rec.get(f"{self.endpoint}_time")
        e = rec.get(f"{self.endpoint}_event")
        if t is None or e is None:
            raise KeyError(
                f"Missing {self.endpoint}_time / {self.endpoint}_event for {pid}"
            )

        out: Dict[str, Any] = {
            "id_": pid,
            "time": torch.tensor(float(t), dtype=torch.float32),
            "event": torch.tensor(float(e), dtype=torch.float32),
        }

        if isinstance(tok, dict) and "cell_sentence" in tok:
            # STATE format: fixed-length sorted gene-index sentence
            out["cell_sentence"] = torch.from_numpy(tok["cell_sentence"].astype(np.int64))
            out["cell_sentence_counts"] = torch.from_numpy(tok["cell_sentence_counts"].astype(np.float32))
        elif isinstance(tok, dict):
            # scGPT / CancerFoundation format: variable-length gene_ids + expressions
            out["gene_ids"] = torch.from_numpy(tok["gene_ids"].astype(np.int64))
            out["expressions"] = torch.from_numpy(tok["expressions"].astype(np.float32))
        elif np.issubdtype(tok.dtype, np.floating):
            # COMPASS format: float32 array (cancer_code + scaled expression)
            out["input_ids"] = torch.from_numpy(tok.astype(np.float32))
        else:
            # BulkRNABert format: int8 array → int64 for nn.Embedding
            out["input_ids"] = torch.from_numpy(tok.astype(np.int64))

        if self.return_metadata:
            for k in self.metadata_keys:
                out[k] = rec.get(k, None)

        return out


# ---------------------------------------------------------------------------
# scGPT collate function
# ---------------------------------------------------------------------------

def scgpt_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    pad_value: float,
    max_length: int = 1200,
    n_bins: int = 51,
) -> Dict[str, Any]:
    """Bin, truncate, and pad a batch of scGPT token dicts.

    Replicates the DataCollator logic from scgpt.data_collator with
    keep_first_n_tokens=1 (preserve the <cls> token at position 0).

    Parameters
    ----------
    batch       : list of items from FinetuneTokensSurvivalDataset
    pad_token_id: vocab ID of the <pad> token
    pad_value   : float pad value for expressions (-2 for scGPT)
    max_length  : maximum sequence length (1200 for scGPT_human)
    n_bins      : number of expression bins (51 for scGPT_human)
    """
    # lazy import so rna_modeling env can still import this module for BulkRNABert
    from scgpt.preprocess import binning

    gene_ids_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []

    for ex in batch:
        genes = ex["gene_ids"]       # (L,) int64
        expr = ex["expressions"]     # (L,) float32, expr[0] = pad_value (<cls>)

        # bin expression values, skipping position 0 (<cls>)
        binned = expr.clone()
        binned[1:] = binning(expr[1:], n_bins=n_bins)

        # truncate: keep <cls> + random subset of remaining genes
        if len(genes) > max_length:
            keep = torch.randperm(len(genes) - 1)[: max_length - 1] + 1
            keep, _ = keep.sort()
            keep = torch.cat([torch.tensor([0], dtype=torch.long), keep])
            genes = genes[keep]
            binned = binned[keep]

        gene_ids_list.append(genes)
        values_list.append(binned)

    # pad all sequences to the longest in this batch (≤ max_length)
    batch_max = max(len(g) for g in gene_ids_list)
    B = len(batch)
    padded_genes = torch.full((B, batch_max), pad_token_id, dtype=torch.long)
    padded_values = torch.full((B, batch_max), pad_value, dtype=torch.float32)

    for i, (g, v) in enumerate(zip(gene_ids_list, values_list)):
        L = len(g)
        padded_genes[i, :L] = g
        padded_values[i, :L] = v

    # True = padding position (matches PyTorch MultiheadAttention convention)
    padding_mask = padded_genes.eq(pad_token_id)

    out: Dict[str, Any] = {
        "gene_ids": padded_genes,
        "values": padded_values,
        "padding_mask": padding_mask,
    }

    # stack survival labels
    out["time"] = torch.stack([ex["time"] for ex in batch])
    out["event"] = torch.stack([ex["event"] for ex in batch])

    # pass-through list fields
    for key in ["id_", "organ", "cancer_type"]:
        if key in batch[0]:
            out[key] = [ex[key] for ex in batch]

    return out


# ---------------------------------------------------------------------------
# CancerFoundation collate function
# ---------------------------------------------------------------------------

def _cf_binning(row: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based expression binning (mirrors CancerFoundation data_collator.py)."""
    if row.max() == 0:
        return np.zeros_like(row, dtype=np.int64)
    if row.min() <= 0:
        nz = np.nonzero(row)[0]
        nz_vals = row[nz]
        bins = np.quantile(nz_vals, np.linspace(0, 1, n_bins - 1))
        left = np.digitize(nz_vals, bins)
        right = np.digitize(nz_vals, bins, right=True)
        rands = np.random.rand(len(nz_vals))
        digits = np.ceil(rands * (right - left) + left).astype(np.int64)
        binned = np.zeros(len(row), dtype=np.int64)
        binned[nz] = digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        left = np.digitize(row, bins)
        right = np.digitize(row, bins, right=True)
        rands = np.random.rand(len(row))
        binned = np.ceil(rands * (right - left) + left).astype(np.int64)
    return binned


def cancerfoundation_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    pad_value: float,
    max_length: int = 1200,
    n_bins: int = 51,
) -> Dict[str, Any]:
    """Bin, truncate, and pad a batch of CancerFoundation token dicts.

    Identical semantics to scgpt_collate_fn but uses an inline binning
    function so the cancerfoundation conda env is not required to have scgpt.

    Parameters
    ----------
    batch        : list of items from FinetuneTokensSurvivalDataset
    pad_token_id : vocab ID of the <pad> token
    pad_value    : float pad value for expressions (-2 for CancerFoundation)
    max_length   : maximum sequence length (1200)
    n_bins       : number of expression bins (51)
    """
    gene_ids_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []

    for ex in batch:
        genes = ex["gene_ids"]       # (L,) int64
        expr = ex["expressions"]     # (L,) float32, expr[0] = pad_value (<cls>)

        # bin expression values, skipping position 0 (<cls>)
        expr_np = expr.numpy()
        binned_np = _cf_binning(expr_np[1:], n_bins=n_bins)
        binned = torch.cat([
            torch.tensor([expr_np[0]], dtype=torch.float32),  # cls pad_value
            torch.from_numpy(binned_np).float(),
        ])

        # truncate: keep <cls> + random subset of remaining genes
        if len(genes) > max_length:
            keep = torch.randperm(len(genes) - 1)[: max_length - 1] + 1
            keep, _ = keep.sort()
            keep = torch.cat([torch.tensor([0], dtype=torch.long), keep])
            genes = genes[keep]
            binned = binned[keep]

        gene_ids_list.append(genes)
        values_list.append(binned)

    # pad to longest in batch (≤ max_length)
    batch_max = max(len(g) for g in gene_ids_list)
    B = len(batch)
    padded_genes = torch.full((B, batch_max), pad_token_id, dtype=torch.long)
    padded_values = torch.full((B, batch_max), pad_value, dtype=torch.float32)

    for i, (g, v) in enumerate(zip(gene_ids_list, values_list)):
        L = len(g)
        padded_genes[i, :L] = g
        padded_values[i, :L] = v

    padding_mask = padded_genes.eq(pad_token_id)

    out: Dict[str, Any] = {
        "gene_ids": padded_genes,
        "values": padded_values,
        "padding_mask": padding_mask,
    }

    out["time"] = torch.stack([ex["time"] for ex in batch])
    out["event"] = torch.stack([ex["event"] for ex in batch])

    for key in ["id_", "organ", "cancer_type"]:
        if key in batch[0]:
            out[key] = [ex[key] for ex in batch]

    return out


# ---------------------------------------------------------------------------
# STATE collate function
# ---------------------------------------------------------------------------

def state_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate STATE cell-sentence tokens.

    STATE tokens have a fixed length (pad_length=2048), so no padding is
    needed — just stack.  Both cell_sentence (int64 gene indices) and
    cell_sentence_counts (float32 expression weights × 100) are pre-computed
    by STATE_tokenize.py.
    """
    out: Dict[str, Any] = {
        "cell_sentence": torch.stack([ex["cell_sentence"] for ex in batch]),         # (B, 2048)
        "cell_sentence_counts": torch.stack([ex["cell_sentence_counts"] for ex in batch]),  # (B, 2048)
        "time": torch.stack([ex["time"] for ex in batch]),
        "event": torch.stack([ex["event"] for ex in batch]),
    }

    for key in ["id_", "organ", "cancer_type"]:
        if key in batch[0]:
            out[key] = [ex[key] for ex in batch]

    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class FinetuneTokensSurvivalDataModule(pl.LightningDataModule):
    """LightningDataModule for FM fine-tuning experiments.

    Loads the main processed data pickle (for survival labels + metadata) and
    a separate tokens pickle produced by a tokenization script, then builds
    train / val / test datasets for a given cross-validation split.

    Parameters
    ----------
    data_path : str
        Path to the main processed pkl ({patient_id: record_dict}).
    tokens_path : str
        Path to the tokens pkl ({patient_id: array or dict}).
    split_folder : str
        Directory containing split_{idx}.pkl files.
    split_idx : int
        Which cross-validation split to use.
    endpoint : str
        Survival endpoint (e.g. "PFS").
    batch_size : int
    num_workers : int
    return_metadata : bool
    metadata_keys : list[str]
    collate_fn : callable, optional
        Custom collate function.  Pass ``scgpt_collate_fn`` (partial-applied
        with pad_token_id / pad_value / max_length) for scGPT.
        Defaults to None (PyTorch default collation, suitable for BulkRNABert).
    """

    def __init__(
        self,
        data_path: str,
        tokens_path: str,
        split_folder: str,
        split_idx: int,
        endpoint: str,
        batch_size: int = 4,
        num_workers: int = 4,
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokens_path = tokens_path
        self.split_folder = split_folder
        self.split_idx = split_idx
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys
        self.collate_fn = collate_fn

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        data: Dict[str, Dict] = load_pickle(self.data_path)
        tokens: Dict[str, Any] = load_pickle(self.tokens_path)

        split_path = os.path.join(self.split_folder, f"split_{self.split_idx}.pkl")
        split = load_pickle(split_path)

        def subset(ids):
            return {p: data[p] for p in ids if p in data}

        train = subset(split["train"])
        val = subset(split["val"])
        test = subset(split["test"])

        n_raw = (len(train), len(val), len(test))

        def keep(rec: Dict) -> bool:
            return (
                rec.get(f"{self.endpoint}_time") is not None
                and rec.get(f"{self.endpoint}_event") is not None
            )

        train = {p: r for p, r in train.items() if keep(r)}
        val = {p: r for p, r in val.items() if keep(r)}
        test = {p: r for p, r in test.items() if keep(r)}

        print(
            "Filtered missing labels / tokens:\n"
            f"  Train: {n_raw[0]} → {len(train)}\n"
            f"  Val:   {n_raw[1]} → {len(val)}\n"
            f"  Test:  {n_raw[2]} → {len(test)}"
        )

        n_in_tokens = sum(1 for p in train if p in tokens)
        print(f"  Patients with tokens: {n_in_tokens} / {len(train)} (train)")

        ds_kwargs = dict(
            tokens=tokens,
            endpoint=self.endpoint,
            return_metadata=self.return_metadata,
            metadata_keys=self.metadata_keys,
        )
        self.train_dataset = FinetuneTokensSurvivalDataset(train, **ds_kwargs)
        self.val_dataset = FinetuneTokensSurvivalDataset(val, **ds_kwargs)
        self.test_dataset = FinetuneTokensSurvivalDataset(test, **ds_kwargs)

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)

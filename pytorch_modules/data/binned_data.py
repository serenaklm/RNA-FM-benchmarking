"""
Binned HVG ablation: data module that discretizes continuous gene expression
values into ordinal bin indices before feeding them to the downstream model.

This isolates the effect of expression discretization (as performed by
foundation models such as BulkRNABert and scGPT) from other factors that
may explain their lower prognostic performance relative to simple baselines.

Bin edges are always fit on the training split only and then applied to
validation and test splits, preventing any data leakage.
"""

from typing import Any, Dict, List, Optional
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_modules.utils.utils import load_pickle
from pytorch_modules.data.data import _fit_hvg_and_apply


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

def fit_bin_edges(
    train_data: Dict[str, Dict],
    n_hvgs: int,
    n_bins: int,
    strategy: str,
) -> np.ndarray:
    """Fit bin edges from training data only.

    Parameters
    ----------
    train_data:
        Dict mapping patient ID -> record dict containing "RNA_seq" key.
    n_hvgs:
        Number of HVGs (top-k slice of RNA_seq).
    n_bins:
        Number of bins.  Edges array has shape (n_hvgs, n_bins + 1).
    strategy:
        "linear"          – equal-width bins *per gene*, spanning each gene's
                            [min, max] across training samples.
        "quantile"        – equal-frequency bins *per gene* (percentile-based).
        "global_linear"   – equal-width bins shared across *all genes*, spanning
                            the global [min, max] of the entire training matrix.
                            Mimics BulkRNABert-style tokenisation.
        "global_quantile" – equal-frequency bins shared across *all genes*,
                            computed on the flattened training matrix.

    Returns
    -------
    bin_edges : np.ndarray, shape (n_hvgs, n_bins + 1), dtype float32
        All rows are identical for the two global strategies.

    Notes
    -----
    Global binning converges to the continuous case as n_bins → ∞.
    Per-sample binning (used by scGPT / CancerFoundation) is a rank
    transformation and does *not* converge regardless of n_bins, because it
    discards between-patient absolute expression differences by design.
    """
    all_x = []
    for rec in train_data.values():
        x = rec.get("RNA_seq")
        if x is None:
            continue
        x = np.asarray(x, dtype=np.float32).squeeze()
        if x.ndim == 1 and len(x) >= n_hvgs:
            all_x.append(x[:n_hvgs])

    if not all_x:
        raise ValueError("No valid training samples found to fit bin edges.")

    all_x = np.stack(all_x, axis=0)  # (n_samples, n_hvgs)

    if strategy == "linear":
        gene_min = all_x.min(axis=0)   # (n_hvgs,)
        gene_max = all_x.max(axis=0)   # (n_hvgs,)
        fracs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        bin_edges = (
            gene_min[:, np.newaxis]
            + (gene_max - gene_min)[:, np.newaxis] * fracs[np.newaxis, :]
        )  # (n_hvgs, n_bins+1)

    elif strategy == "quantile":
        quantile_vals = np.linspace(0.0, 100.0, n_bins + 1)
        bin_edges = np.percentile(all_x, quantile_vals, axis=0).T  # (n_hvgs, n_bins+1)
        bin_edges = bin_edges.astype(np.float32)

    elif strategy == "global_linear":
        global_min = float(all_x.min())
        global_max = float(all_x.max())
        edges = np.linspace(global_min, global_max, n_bins + 1, dtype=np.float32)
        bin_edges = np.tile(edges, (n_hvgs, 1))  # same edges for every gene

    elif strategy == "global_quantile":
        quantile_vals = np.linspace(0.0, 100.0, n_bins + 1)
        edges = np.percentile(all_x.ravel(), quantile_vals).astype(np.float32)
        bin_edges = np.tile(edges, (n_hvgs, 1))  # same edges for every gene

    else:
        raise ValueError(
            f"Unknown bin strategy '{strategy}'. "
            "Use 'linear', 'quantile', 'global_linear', or 'global_quantile'."
        )

    return bin_edges


def apply_binning(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous expression values to normalised bin indices in [0, 1].

    Parameters
    ----------
    x : (n_hvgs,) float32 array of expression values.
    bin_edges : (n_hvgs, n_bins+1) float32 array of per-gene bin edges.

    Returns
    -------
    indices : (n_hvgs,) float32 array of bin indices normalised to [0, 1].
        Normalising by (n_bins - 1) keeps the input scale constant across
        different n_bins values, preventing gradient explosion at large n_bins
        and ensuring the comparison across n_bins is not confounded by scale.
    """
    n_bins = bin_edges.shape[1] - 1
    # Vectorised: for each gene g, count how many edges x[g] is >= to,
    # then subtract 1 for 0-based indexing.
    #   x[:, None] >= bin_edges  →  (n_hvgs, n_bins+1) bool
    #   .sum(axis=1)             →  (n_hvgs,) int, range [0, n_bins+1]
    indices = (x[:, np.newaxis] >= bin_edges).sum(axis=1).astype(np.float32) - 1.0
    indices = np.clip(indices, 0.0, float(n_bins - 1))
    return indices / float(max(n_bins - 1, 1))  # normalise to [0, 1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinnedHVGSurvivalDataset(Dataset):
    """Survival dataset that replaces continuous HVG expression with ordinal
    bin indices.

    For per-gene / global strategies, bin_edges must be pre-fit on training
    data and passed in.  For the per_sample strategy, bin_edges is ignored
    and binning is computed on-the-fly within each patient's expression vector.
    """

    def __init__(
        self,
        data: Dict[str, Dict],
        endpoint: str,
        n_hvgs: int,
        bin_strategy: str,
        n_bins: int,
        bin_edges: Optional[np.ndarray],
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data = data
        self.ids = list(data.keys())
        self.endpoint = endpoint
        self.n_hvgs = n_hvgs
        self.bin_strategy = bin_strategy
        self.n_bins = n_bins
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys or ["organ", "cancer_type"]

        # Pre-compute binned expression for every sample once at construction
        # time.  Doing this in __getitem__ would repeat the work every epoch,
        # which is O(n_hvgs × n_bins) per sample and becomes the bottleneck at
        # large n_bins (e.g. (10000, 1025) boolean array per sample per epoch).
        scale = float(max(n_bins - 1, 1))
        self._binned: Dict[str, np.ndarray] = {}
        for pid, rec in data.items():
            x = rec.get("RNA_seq")
            if x is None:
                continue
            x = np.asarray(x, dtype=np.float32).squeeze()[:n_hvgs]

            if bin_strategy == "per_sample_quantile":
                ranks = np.argsort(np.argsort(x))
                bx = np.floor(ranks / n_hvgs * n_bins).astype(np.float32)
                bx = np.clip(bx, 0.0, float(n_bins - 1)) / scale
            elif bin_strategy == "per_sample_linear":
                x_min, x_max = float(x.min()), float(x.max())
                if x_max > x_min:
                    bx = np.floor((x - x_min) / (x_max - x_min) * n_bins).astype(np.float32)
                else:
                    bx = np.zeros(n_hvgs, dtype=np.float32)
                bx = np.clip(bx, 0.0, float(n_bins - 1)) / scale
            else:
                bx = apply_binning(x, bin_edges)

            self._binned[pid] = bx

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.ids[idx]
        rec = self.data[pid]

        x = self._binned.get(pid)
        if x is None:
            raise KeyError(f"Missing RNA_seq for sample {pid}")

        t = rec.get(f"{self.endpoint}_time")
        e = rec.get(f"{self.endpoint}_event")
        if t is None or e is None:
            raise KeyError(f"Missing survival labels for {pid}: "
                           f"{self.endpoint}_time / {self.endpoint}_event")

        out: Dict[str, Any] = {
            "id_":   pid,
            "x_rna": torch.from_numpy(x),
            "time":  torch.tensor(float(t), dtype=torch.float32),
            "event": torch.tensor(float(e), dtype=torch.float32),
        }

        if self.return_metadata:
            for k in self.metadata_keys:
                out[k] = rec.get(k, None)

        return out


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class BinnedHVGSurvivalDataModule(pl.LightningDataModule):
    """LightningDataModule for the binned-HVG ablation experiment.

    Fits per-gene bin edges on the training split only and applies them
    uniformly to train / val / test, so there is no data leakage.
    """

    def __init__(
        self,
        data_path: str,
        split_folder: str,
        split_idx: int,
        endpoint: str,
        n_HVGs: int,
        n_bins: int,
        bin_strategy: str,
        batch_size: int = 128,
        num_workers: int = 4,
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.split_folder = split_folder
        self.split_idx = split_idx
        self.endpoint = endpoint
        self.n_hvgs = n_HVGs
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys
        self.rna_dim: Optional[int] = None
        self._setup_done = False

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        if self._setup_done:
            return
        self._setup_done = True

        data: Dict[str, Dict] = load_pickle(self.data_path)

        split_path = os.path.join(self.split_folder, f"split_{self.split_idx}.pkl")
        split = load_pickle(split_path)

        def subset(ids):
            return {p: data[p] for p in ids if p in data}

        train = subset(split["train"])
        val   = subset(split["val"])
        test  = subset(split["test"])

        n_train_raw, n_val_raw, n_test_raw = len(train), len(val), len(test)

        def keep(rec: Dict) -> bool:
            x = rec.get("RNA_seq")
            if x is None or len(np.asarray(x).squeeze()) < self.n_hvgs:
                return False
            return (
                rec.get(f"{self.endpoint}_time") is not None
                and rec.get(f"{self.endpoint}_event") is not None
            )

        train = {p: r for p, r in train.items() if keep(r)}
        val   = {p: r for p, r in val.items()   if keep(r)}
        test  = {p: r for p, r in test.items()  if keep(r)}

        print(
            "Filtered missing labels / representations:\n"
            f"  Train: {n_train_raw} → {len(train)}\n"
            f"  Val:   {n_val_raw} → {len(val)}\n"
            f"  Test:  {n_test_raw} → {len(test)}"
        )

        # Select HVGs using training-set variance only; overwrite RNA_seq in all splits.
        # Shallow-copy records first so we do not mutate the master pkl dict.
        train = {p: dict(r) for p, r in train.items()}
        val   = {p: dict(r) for p, r in val.items()}
        test  = {p: dict(r) for p, r in test.items()}
        _fit_hvg_and_apply(self.n_hvgs, train, val, test)

        if self.bin_strategy in ("per_sample_quantile", "per_sample_linear"):
            bin_edges = None
            print(
                f"Per-sample binning ({self.bin_strategy}) with {self.n_bins} bins "
                f"(no edge fitting — normalised within each patient)."
            )
        else:
            print(
                f"Fitting {self.n_bins} {self.bin_strategy} bin edges on "
                f"{len(train)} training samples for {self.n_hvgs} HVGs..."
            )
            bin_edges = fit_bin_edges(train, self.n_hvgs, self.n_bins, self.bin_strategy)
            print(f"Bin edges shape: {bin_edges.shape}")

        self.rna_dim = self.n_hvgs

        ds_kwargs = dict(
            endpoint=self.endpoint,
            n_hvgs=self.n_hvgs,
            bin_strategy=self.bin_strategy,
            n_bins=self.n_bins,
            bin_edges=bin_edges,
            return_metadata=self.return_metadata,
            metadata_keys=self.metadata_keys,
        )
        self.train_dataset = BinnedHVGSurvivalDataset(train, **ds_kwargs)
        self.val_dataset   = BinnedHVGSurvivalDataset(val,   **ds_kwargs)
        self.test_dataset  = BinnedHVGSurvivalDataset(test,  **ds_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True,
            drop_last=False, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False,
            drop_last=False, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False,
            drop_last=False, pin_memory=True,
        )

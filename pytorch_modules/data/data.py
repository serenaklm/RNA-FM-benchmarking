from typing import Any, Dict, Optional, List
import os

import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.decomposition import PCA as _PCA
from sklearn.model_selection import train_test_split

from pytorch_modules.utils.utils import load_pickle, load_h5
from torch.utils.data import Dataset, DataLoader
import pickle

def _to_float_tensor(x) -> torch.Tensor:
    # Handles numpy arrays / lists / torch tensors
    if x is None:
        raise ValueError("Got None where an array was expected.")
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _fit_pca_and_apply(n_components: int, train: dict, val: dict, test: dict) -> None:
    """Fit PCA on training RNA_seq only; overwrite PCA_{n_components} in all splits.

    Records in all three dicts must already be shallow copies (not shared with
    the master pkl) so the injected key does not mutate the source data.
    """
    import time
    key = f"PCA_{n_components}"
    train_ids = list(train.keys())
    train_rna = np.stack([
        np.asarray(train[p]["RNA_seq"], dtype=np.float32).ravel()
        for p in train_ids
    ])
    n_actual = min(n_components, train_rna.shape[0] - 1, train_rna.shape[1])
    if n_actual < n_components:
        print(f"[PCA] Capping components: {n_components} → {n_actual} "
              f"(n_train={train_rna.shape[0]}, n_genes={train_rna.shape[1]})", flush=True)
    pca = _PCA(n_components=n_actual, random_state=0)
    print(f"[PCA] Starting fit: PCA({n_actual}) on matrix {train_rna.shape} ...", flush=True)
    t0 = time.time()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api="blas"):
        pca.fit(train_rna)
    print(f"[PCA] Fit done in {time.time()-t0:.1f}s. Transforming splits ...", flush=True)
    for split_dict in (train, val, test):
        if not split_dict:
            continue
        ids = list(split_dict.keys())
        rna = np.stack([
            np.asarray(split_dict[p]["RNA_seq"], dtype=np.float32).ravel()
            for p in ids
        ])
        Z = pca.transform(rna).astype(np.float32)
        for i, pid in enumerate(ids):
            split_dict[pid][key] = Z[i]
    print(f"[PCA] All splits transformed. Total time: {time.time()-t0:.1f}s", flush=True)


def _fit_hvg_and_apply(n_hvgs: int, train: dict, val: dict, test: dict) -> None:
    """Select top-n_hvgs genes by training-set variance; overwrite RNA_seq in all splits.

    Records in all three dicts must already be shallow copies so mutations do
    not propagate to the master pkl.
    """
    train_rna = np.stack([
        np.asarray(r["RNA_seq"], dtype=np.float32).ravel()
        for r in train.values()
    ])
    train_var = np.var(train_rna, axis=0)
    hvg_idx = np.argsort(train_var)[::-1][:n_hvgs]
    print(f"[HVG] Selected top-{n_hvgs} HVGs by training-set variance.")
    for split_dict in (train, val, test):
        for rec in split_dict.values():
            rec["RNA_seq"] = np.asarray(rec["RNA_seq"], dtype=np.float32).ravel()[hvg_idx]


# Generic code
class RNAOnlySurvivalDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Dict],
        endpoint: str,
        repr_key: str = "RNA_seq",
        n_hvgs: Optional[int] = None,
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,  # e.g. ["organ", "cancer_type"]
    ):
        super().__init__()
        self.data = data
        self.ids = list(data.keys())
        self.endpoint = endpoint
        self.repr_key = repr_key
        self.n_hvgs = n_hvgs
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys or ["organ", "cancer_type"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.ids[idx]
        rec = self.data[pid]

        x = rec.get(self.repr_key, None)
        if x is None:
            raise KeyError(f"Missing {self.repr_key} for {pid}")

        # HVG slice only if raw RNA_seq
        if self.repr_key == "RNA_seq":
            if self.n_hvgs is not None:
                x = x[: self.n_hvgs]

        x = np.asarray(x).squeeze()
        assert x.ndim == 1, f"{pid}: {self.repr_key} has shape {x.shape}, expected (D,)"

        t = rec.get(f"{self.endpoint}_time", None)
        e = rec.get(f"{self.endpoint}_event", None)
        if t is None or e is None:
            raise KeyError(f"Missing survival labels for {pid}: {self.endpoint}_time/event")

        out = {
            "id_": pid,
            "x_rna": _to_float_tensor(x),
            "time": torch.tensor(float(t), dtype=torch.float32),
            "event": torch.tensor(float(e), dtype=torch.float32),
        }

        if self.return_metadata:
            for k in self.metadata_keys:
                out[k] = rec.get(k, None)

        return out

class SplitRNAOnlySurvivalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        split_folder: str,
        split_idx: int,
        endpoint: str,
        rna_embedding_key: str = "RNA_seq",
        n_HVGs: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        filter_fn=None,  # optional: fn(rec)->bool
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.split_folder = split_folder
        self.split_idx = split_idx
        self.endpoint = endpoint
        self.repr_key = rna_embedding_key
        self.n_hvgs = n_HVGs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter_fn = filter_fn
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys

        self.rna_dim: Optional[int] = None
        self._setup_done = False

    def setup(self, stage: Optional[str] = None):
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

        def _has_surv(rec):
            return (
                rec.get(f"{self.endpoint}_time") is not None
                and rec.get(f"{self.endpoint}_event") is not None
            )

        def _has_repr(rec):
            # PCA embeddings are computed from RNA_seq at setup time
            if self.repr_key.startswith("PCA_"):
                return rec.get("RNA_seq") is not None
            return rec.get(self.repr_key) is not None

        def keep(pid, rec):
            ok = _has_repr(rec) and _has_surv(rec)
            if not ok:
                return False
            if self.filter_fn is not None:
                return bool(self.filter_fn(pid, rec))  # <-- needs pid here
            return True

        train = {p: r for p, r in train.items() if keep(p, r)}
        val   = {p: r for p, r in val.items()   if keep(p, r)}
        test  = {p: r for p, r in test.items()  if keep(p, r)}

        print(
            "Filtered missing labels / representations:\n"
            f"  Train: {n_train_raw} to {len(train)}  (dropped {n_train_raw - len(train)})\n"
            f"  Val:   {n_val_raw} to {len(val)}  (dropped {n_val_raw - len(val)})\n"
            f"  Test:  {n_test_raw} to {len(test)}  (dropped {n_test_raw - len(test)})"
        )

        # Fit PCA or HVG selection on training data only, then apply to all splits.
        # Shallow-copy records first so we do not mutate the master pkl dict.
        n_hvgs_for_datasets = self.n_hvgs
        if self.repr_key.startswith("PCA_") or (self.repr_key == "RNA_seq" and self.n_hvgs is not None):
            train = {p: dict(r) for p, r in train.items()}
            val   = {p: dict(r) for p, r in val.items()}
            test  = {p: dict(r) for p, r in test.items()}
            if self.repr_key.startswith("PCA_"):
                n_components = int(self.repr_key.split("_")[1])
                _fit_pca_and_apply(n_components, train, val, test)
            else:
                _fit_hvg_and_apply(self.n_hvgs, train, val, test)
                # RNA_seq is now pre-sliced; tell Dataset not to re-slice.
                n_hvgs_for_datasets = None

        # Infer dim
        any_pid = next(iter(train.keys()))
        x0 = train[any_pid][self.repr_key]
        if self.repr_key == "RNA_seq":
            if self.n_hvgs is None:
                self.rna_dim = int(np.asarray(x0).squeeze().shape[-1])
            else:
                self.rna_dim = int(self.n_hvgs)
        else:
            self.rna_dim = int(np.asarray(x0).squeeze().shape[-1])
        print(f"RNA representation key: {self.repr_key}, dim={self.rna_dim}")

        ds_kwargs = dict(
            endpoint=self.endpoint,
            repr_key=self.repr_key,
            n_hvgs=n_hvgs_for_datasets,
            return_metadata=self.return_metadata,
            metadata_keys=self.metadata_keys,
        )
        self.train_dataset = RNAOnlySurvivalDataset(train, **ds_kwargs)
        self.val_dataset   = RNAOnlySurvivalDataset(val,   **ds_kwargs)
        self.test_dataset  = RNAOnlySurvivalDataset(test,  **ds_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          drop_last=False, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          drop_last=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          drop_last=False, pin_memory=False)

class TCGADataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4):
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            batch_size=batch_size,
            num_workers=num_workers,
        )

COMPASS_MAPPING_PATHS = {
    "CPTAC":   "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/CPTAC/id_to_tcga_mapping.pkl",
    "ICGC":    "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/ICGC/id_to_tcga_mapping.pkl",
    "Hartwig": "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/Hartwig/id_to_tcga_mapping.pkl",
}

def _make_compass_filter(mapping_path: str):
    valid_ids = set(load_pickle(mapping_path).keys())
    return lambda pid, rec: pid in valid_ids


class CPTACDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4):
        compass_filter = _make_compass_filter(COMPASS_MAPPING_PATHS["CPTAC"]) \
                         if rna_embedding_key == "COMPASS" else None
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            filter_fn=compass_filter,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class HartwigDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4):
        compass_filter = _make_compass_filter(COMPASS_MAPPING_PATHS["Hartwig"]) \
                         if rna_embedding_key == "COMPASS" else None
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            filter_fn=compass_filter,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class ICGCDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4):
        compass_filter = _make_compass_filter(COMPASS_MAPPING_PATHS["ICGC"]) \
                         if rna_embedding_key == "COMPASS" else None
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            filter_fn=compass_filter,
            batch_size=batch_size,
            num_workers=num_workers,
        )

class TARGET_AMLDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4):
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
class CALGBDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4):
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            batch_size=batch_size,
            num_workers=num_workers,
        )

class RNAOnlyResponseDataset(Dataset):
    """
    Binary classification dataset for Hartwig response (0/1).
    Uses the same representation logic as RNAOnlySurvivalDataset.
    """
    def __init__(
        self,
        data: Dict[str, Dict],
        repr_key: str = "RNA_seq",
        n_hvgs: Optional[int] = None,
        label_key: str = "response",
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data = data
        self.ids = list(data.keys())
        self.repr_key = repr_key
        self.n_hvgs = n_hvgs
        self.label_key = label_key
        self.return_metadata = return_metadata
        # Default to standard keys if none provided
        self.metadata_keys = metadata_keys or ["organ", "cancer_type"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.ids[idx]
        rec = self.data[sid]

        # 1. Handle Features (RNA)
        x = rec.get(self.repr_key, None)
        if x is None:
            raise KeyError(f"Missing {self.repr_key} for {sid}")

        if self.repr_key == "RNA_seq":
            if self.n_hvgs is not None:
                x = x[: self.n_hvgs]

        x = np.asarray(x).squeeze()
        assert x.ndim == 1, f"{sid}: {self.repr_key} has shape {x.shape}, expected (D,)"

        # 2. Handle Label
        y = rec.get(self.label_key, None)
        if y is None:
            raise KeyError(f"Missing {self.label_key} for {sid}")

        out = {
            "id_": sid,
            "x_rna": _to_float_tensor(x),
            "y": torch.tensor(int(y), dtype=torch.long),
        }

        # 3. Handle Metadata (The likely source of the error)
        if self.return_metadata:
            for k in self.metadata_keys:
                val = rec.get(k, None)
                
                # Check 1: Handle None values (PyTorch collate hates None)
                if val is None:
                    out[k] = "Unknown"
                
                # Check 2: Handle Lists (PyTorch collate requires equal length lists)
                elif isinstance(val, list):
                    # Join list into a single string to ensure consistency
                    out[k] = "|".join(map(str, val))
                
                else:
                    out[k] = val

        return out

class SplitRNAOnlyResponseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        split_folder: str,
        split_idx: int,
        rna_embedding_key: str = "RNA_seq",
        n_HVGs: Optional[int] = None,
        label_key: str = "response",
        batch_size: int = 64,
        num_workers: int = 4,
        filter_fn=None,
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
        use_reference_hvg: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.split_folder = split_folder
        self.split_idx = split_idx
        self.repr_key = rna_embedding_key
        self.n_hvgs = n_HVGs
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter_fn = filter_fn
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys
        # When True, skip train-only HVG reselection and use the reference gene
        # ordering already baked into RNA_seq (e.g. TCGA variance for ICI cohorts).
        self.use_reference_hvg = use_reference_hvg

        self.rna_dim: Optional[int] = None
        self._setup_done = False

    def setup(self, stage: Optional[str] = None):
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

        def _has_repr(rec):
            # PCA embeddings are computed from RNA_seq at setup time
            if self.repr_key.startswith("PCA_"):
                return rec.get("RNA_seq") is not None
            return rec.get(self.repr_key) is not None

        def _has_label(rec):
            return rec.get(self.label_key) is not None

        def keep(pid, rec):
            ok = _has_repr(rec) and _has_label(rec)
            if not ok:
                return False
            if self.filter_fn is not None:
                return bool(self.filter_fn(pid, rec))  # <-- needs pid here
            return True

        train = {p: r for p, r in train.items() if keep(p, r)}
        val   = {p: r for p, r in val.items()   if keep(p, r)}
        test  = {p: r for p, r in test.items()  if keep(p, r)}

        print(
            "Filtered missing labels / representations:\n"
            f"  Train: {n_train_raw} to {len(train)}  (dropped {n_train_raw - len(train)})\n"
            f"  Val:   {n_val_raw} to {len(val)}  (dropped {n_val_raw - len(val)})\n"
            f"  Test:  {n_test_raw} to {len(test)}  (dropped {n_test_raw - len(test)})"
        )

        # Fit PCA or HVG selection on training data only, then apply to all splits.
        # Shallow-copy records first so we do not mutate the master pkl dict.
        n_hvgs_for_datasets = self.n_hvgs
        needs_pca = self.repr_key.startswith("PCA_")
        needs_hvg = (self.repr_key == "RNA_seq" and self.n_hvgs is not None
                     and not self.use_reference_hvg)
        if needs_pca or needs_hvg:
            train = {p: dict(r) for p, r in train.items()}
            val   = {p: dict(r) for p, r in val.items()}
            test  = {p: dict(r) for p, r in test.items()}
            if needs_pca:
                n_components = int(self.repr_key.split("_")[1])
                _fit_pca_and_apply(n_components, train, val, test)
            else:
                _fit_hvg_and_apply(self.n_hvgs, train, val, test)
                # RNA_seq is now pre-sliced; tell Dataset not to re-slice.
                n_hvgs_for_datasets = None

        any_id = next(iter(train.keys()))
        x0 = train[any_id][self.repr_key]
        if self.repr_key == "RNA_seq":
            if self.n_hvgs is None:
                self.rna_dim = int(np.asarray(x0).squeeze().shape[-1])
            else:
                self.rna_dim = int(self.n_hvgs)
        else:
            self.rna_dim = int(np.asarray(x0).squeeze().shape[-1])
        print(f"RNA representation key: {self.repr_key}, dim={self.rna_dim}")

        ds_kwargs = dict(
            repr_key=self.repr_key,
            n_hvgs=n_hvgs_for_datasets,
            label_key=self.label_key,
            return_metadata=self.return_metadata,
            metadata_keys=self.metadata_keys,
        )
        self.train_dataset = RNAOnlyResponseDataset(train, **ds_kwargs)
        self.val_dataset   = RNAOnlyResponseDataset(val,   **ds_kwargs)
        self.test_dataset  = RNAOnlyResponseDataset(test,  **ds_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          drop_last=False, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          drop_last=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          drop_last=False, pin_memory=False)

class HartwigDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4,
                 return_metadata: bool = True, metadata_keys: Optional[List[str]] = None):
        compass_filter = _make_compass_filter(COMPASS_MAPPING_PATHS["Hartwig"]) \
                         if rna_embedding_key == "COMPASS" else None
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            filter_fn=compass_filter,
            batch_size=batch_size,
            num_workers=num_workers,
            return_metadata=return_metadata,
            metadata_keys=metadata_keys,
        )


class HartwigResponseDataModule(SplitRNAOnlyResponseDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4,
                 label_key: str = "response",
                 return_metadata: bool = True, metadata_keys: Optional[List[str]] = None):
        compass_filter = _make_compass_filter(COMPASS_MAPPING_PATHS["Hartwig"]) \
                         if rna_embedding_key == "COMPASS" else None
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            filter_fn=compass_filter,
            label_key=label_key,
            batch_size=batch_size,
            num_workers=num_workers,
            return_metadata=return_metadata,
            metadata_keys=metadata_keys,
        )
    
'''  
class HartwigDataModule(SplitRNAOnlySurvivalDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 endpoint: str, rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4,
                 return_metadata: bool = True, metadata_keys: Optional[List[str]] = None):
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            endpoint=endpoint,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            batch_size=batch_size,
            num_workers=num_workers,
            return_metadata=return_metadata,
            metadata_keys=metadata_keys,
        )

class HartwigResponseDataModule(SplitRNAOnlyResponseDataModule):
    def __init__(self, data_path: str, split_folder: str, split_idx: int,
                 rna_embedding_key: str, n_HVGs: Optional[int],
                 batch_size: int = 64, num_workers: int = 4,
                 label_key: str = "response",
                 return_metadata: bool = True, metadata_keys: Optional[List[str]] = None):
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            label_key=label_key,
            batch_size=batch_size,
            num_workers=num_workers,
            return_metadata=return_metadata,
            metadata_keys=metadata_keys,
        )
'''
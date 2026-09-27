"""
pytorch_modules/data/ici_data.py

Generic DataModule for ICI response prediction.

Works for all three evaluation paradigms (within-cohort, LOCO, transfer) because
the paradigm is entirely encoded in the split file — the DataModule just reads
whatever train/val/test IDs are given.

The processed pkl is expected to contain records with at minimum:
    response   : int  (1 = responder, 0 = non-responder)
    RNA_seq    : array aligned to the TCGA gene vocab
    <FM keys>  : BulkFormer, scGPT, COMPASS, etc.
    cohort     : str   (used for per-cohort evaluation slices)
    cancer_type: str
    organ      : str
"""

from typing import Optional, List
from pytorch_modules.data.data import SplitRNAOnlyResponseDataModule


class ICIResponseDataModule(SplitRNAOnlyResponseDataModule):
    """
    DataModule for ICI cohort response prediction.

    Thin wrapper over SplitRNAOnlyResponseDataModule that:
      - sets label_key="response" (matching the field written by get_ICI_embeddings.py)
      - exposes cohort in metadata so eval can slice per-cohort metrics
    """

    def __init__(
        self,
        data_path: str,
        split_folder: str,
        split_idx: int,
        rna_embedding_key: str = "RNA_seq",
        n_HVGs: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        return_metadata: bool = True,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__(
            data_path=data_path,
            split_folder=split_folder,
            split_idx=split_idx,
            rna_embedding_key=rna_embedding_key,
            n_HVGs=n_HVGs,
            label_key="response",
            batch_size=batch_size,
            num_workers=num_workers,
            return_metadata=return_metadata,
            # include cohort so eval_ICI_models.py can compute per-cohort metrics
            metadata_keys=metadata_keys or ["cohort", "cancer_type", "organ"],
            # ICI RNA_seq is aligned to TCGA variance ordering, which serves as a
            # stable external reference (TCGA is not part of the ICI split).
            use_reference_hvg=True,
        )

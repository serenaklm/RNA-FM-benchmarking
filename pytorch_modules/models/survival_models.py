# pytorch_modules/models/survival_models.py

import pytorch_lightning as pl

import torch
import torch.nn as nn

from pytorch_modules.nn.residual_MLP import ResidualMLP
from pytorch_modules.nn.MLP import MLP

from pytorch_modules.losses.elastic_loss import elastic_loss
from pytorch_modules.losses.cindex_loss import CIndexLoss
from pytorch_modules.losses.cox_loss import cox_ph_loss
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from typing import Optional


_NONLINEARITIES = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,        # aka Swish
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
}


class RNA_SurvivalModel(pl.LightningModule):
    """
    Unified RNA survival LightningModule.

    Supports:
      - architecture: "Linear", "MLP", "ResidualMLP"
      - loss_name:   "Cox", "CIndex"

    Input expected in batch:
      - batch["x_rna"] : (B, D)
      - batch["time"]  : (B,)
      - batch["event"] : (B,) float {0,1}
      - batch["id_"], batch["organ"] (optional, for predict_step parity)

    Output:
      - risk score (B,)
    """

    def _act(self) -> nn.Module:
        if self.non_linearity not in _NONLINEARITIES:
            raise ValueError(
                f"non_linearity must be one of {list(_NONLINEARITIES.keys())}, "
                f"got: {self.non_linearity}"
            )
        return _NONLINEARITIES[self.non_linearity]()

    def __init__(
        self,
        arch: str = "ResidualMLP",          # "Linear" | "MLP" | "ResidualMLP"
        loss_name: str = "CIndex",          # "Cox" | "CIndex"
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        rna_dim: int = 17490,
        output_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 3,
        non_linearity: str = "ReLU",
        dropout_rate: float = 0.2,
        penalty_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.arch = arch
        self.loss_name = loss_name

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.rna_dim = int(rna_dim)

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.non_linearity = non_linearity
        self.dropout_rate = float(dropout_rate)

        self.penalty_weight = float(penalty_weight)

        # Activation (for nonlinear nets)
        self.act = self._act()

        # -------------------------
        # Build model backbone
        # -------------------------
        if self.arch == "Linear":
            # Classical Cox / linear head
            self.model = nn.Linear(self.rna_dim, output_dim, bias=False)

        elif self.arch == "MLP":
            # Assumes your MLP signature is: MLP(num_layers, input_dim, output_dim, hidden_dim, act, dropout_rate)
            self.model = MLP(self.num_layers, self.rna_dim, output_dim, self.hidden_dim, self.act, self.dropout_rate)

        elif self.arch == "ResidualMLP":
            self.model = ResidualMLP(self.num_layers, self.rna_dim, output_dim, self.hidden_dim, self.act, self.dropout_rate)

        else:
            raise ValueError(f"Unknown arch: {self.arch}. Must be one of ['Linear','MLP','ResidualMLP'].")

        # -------------------------
        # Loss selection
        # -------------------------
        if self.loss_name == "Cox":
            self.loss_fn = cox_ph_loss
        elif self.loss_name == "CIndex":
            self.loss_fn = CIndexLoss()
        else:
            raise ValueError(f"Unknown loss_name: {self.loss_name}. Must be one of ['Cox','CIndex'].")

    def forward(self, batch):
        rna = batch.get("x_rna", batch.get("RNA_seq"))
        risk = self.model(rna).squeeze(-1)
        return risk

    def training_step(self, batch, batch_idx):
        risk = self(batch)
        pred_loss = self.loss_fn(risk, batch["time"], batch["event"])

        if self.penalty_weight > 0:
            penalty_l1, penalty_l2 = elastic_loss(self.model)
            loss = pred_loss + self.penalty_weight * (penalty_l1 + penalty_l2)
            self.log("train/l1_loss", penalty_l1, prog_bar=False, on_epoch=True)
            self.log("train/l2_loss", penalty_l2, prog_bar=False, on_epoch=True)
        else:
            loss = pred_loss

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        risk = self(batch)
        pred_loss = self.loss_fn(risk, batch["time"], batch["event"])

        self.log("val/loss", pred_loss, prog_bar=True, on_epoch=True)
        # Keep this for checkpointing / file-saving logic
        self.log("val_loss", pred_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        id_ = batch.get("id_", None)
        time = batch["time"].detach().cpu().numpy().tolist()
        event = batch["event"].detach().cpu().numpy().tolist()
        organ = batch.get("organ", None)

        risk = self(batch).detach().cpu().numpy().tolist()
        #return id_, risk, time, event, organ
        
        cancer_type = batch.get("cancer_type", None)
        return id_, risk, time, event, organ, cancer_type

    def on_after_backward(self):
        self.log("train/grad_norm", self.gradient_norm, prog_bar=False)
        self.log("train/param_norm", self.parameter_norm, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @property
    def gradient_norm(self) -> torch.Tensor:
        parameters = filter(lambda p: p.requires_grad and p.grad is not None, self.parameters())
        # If no grads (rare), return 0.0 safely
        grads = [param.grad.norm(p=2) ** 2 for param in parameters]
        if len(grads) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(grads).sum().sqrt()

    @property
    def parameter_norm(self) -> torch.Tensor:
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = [param.norm(p=2) ** 2 for param in parameters]
        if len(params) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(params).sum().sqrt()

class RNA_ResponseModel(pl.LightningModule):
    """
    Binary response prediction LightningModule (0/1).

    Supports:
      - arch: "Linear", "MLP", "ResidualMLP"
    Expects batch:
      - batch["x_rna"] : (B, D)
      - batch["y"]     : (B,) int {0,1}  (from RNAOnlyResponseDataset)
    Outputs:
      - logits (B,)
    """

    def _act(self) -> nn.Module:
        if self.non_linearity not in _NONLINEARITIES:
            raise ValueError(
                f"non_linearity must be one of {list(_NONLINEARITIES.keys())}, "
                f"got: {self.non_linearity}"
            )
        return _NONLINEARITIES[self.non_linearity]()

    def __init__(
        self,
        arch: str = "ResidualMLP",          # "Linear" | "MLP" | "ResidualMLP"
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        rna_dim: int = 17490,
        hidden_dim: int = 512,
        num_layers: int = 3,
        non_linearity: str = "ReLU",
        dropout_rate: float = 0.2,
        penalty_weight: float = 0.0,
        pos_weight: Optional[float] = None,  # for class imbalance (optional)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.arch = arch
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.rna_dim = int(rna_dim)

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.non_linearity = non_linearity
        self.dropout_rate = float(dropout_rate)

        self.penalty_weight = float(penalty_weight)

        self.act = self._act()

        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()

        # Backbone (same as survival)
        if self.arch == "Linear":
            self.model = nn.Linear(self.rna_dim, 1, bias=True)
        elif self.arch == "MLP":
            self.model = MLP(self.num_layers, self.rna_dim, 1, self.hidden_dim, self.act, self.dropout_rate)
        elif self.arch == "ResidualMLP":
            self.model = ResidualMLP(self.num_layers, self.rna_dim, 1, self.hidden_dim, self.act, self.dropout_rate)
        else:
            raise ValueError(f"Unknown arch: {self.arch}. Must be one of ['Linear','MLP','ResidualMLP'].")

        # Loss
        if pos_weight is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]))

    def forward(self, batch):
        rna = batch.get("x_rna", batch.get("RNA_seq"))
        logits = self.model(rna).squeeze(-1)  # (B,)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch["y"].float()
        pred_loss = self.loss_fn(logits, y)

        if self.penalty_weight > 0:
            penalty_l1, penalty_l2 = elastic_loss(self.model)
            loss = pred_loss + self.penalty_weight * (penalty_l1 + penalty_l2)
            self.log("train/l1_loss", penalty_l1, prog_bar=False, on_epoch=True)
            self.log("train/l2_loss", penalty_l2, prog_bar=False, on_epoch=True)
        else:
            loss = pred_loss

        # Simple accuracy (threshold 0 on logits)
        preds = (logits > 0).long()
        acc = (preds == batch["y"]).float().mean()

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch["y"].float()
        loss = self.loss_fn(logits, y)

        preds = (logits > 0).long()
        acc = (preds == batch["y"]).float().mean()

        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, batch["y"].long())
        self.val_auprc.update(probs, batch["y"].long())

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/auprc", self.val_auprc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    """
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        id_ = batch.get("id_", None)
        logits = self(batch).detach().cpu().numpy().tolist()
        probs = torch.sigmoid(torch.tensor(logits)).numpy().tolist()
        y = batch.get("y", None)
        y = y.detach().cpu().numpy().tolist() if y is not None else None
        return id_, probs, logits, y
    """
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        id_ = batch.get("id_", None)
        organ = batch.get("organ", None)
        cohort = batch.get("cohort", None)
        cancer_type = batch.get("cancer_type", None)
        y = batch["y"].detach().cpu().numpy().tolist()

        logits = self(batch).detach().cpu().numpy().tolist()

        return id_, logits, y, organ, cancer_type, cohort


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


"""
Fine-tuning models for transcriptomic foundation models.

Each FM subclass wraps the backbone with LoRA (via peft) and attaches
a linear survival head.  Add a new FM by subclassing FMFinetuneSurvivalModel
and registering it in FM_REGISTRY.

Memory note — BulkRNABert
--------------------------
BulkRNABert uses standard O(N²) self-attention over N=19,062 genes.
At bf16, one attention map = (B, n_heads, N, N) × 2 bytes:
  B=4, 8 heads → ~23 GB.
To avoid OOM we run layers 0..N-2 under torch.no_grad() and only compute
gradients for the last layer (≈ 5.8 GB peak at B=1).

Memory note — scGPT
--------------------
scGPT processes at most 1,200 HVGs.  At bf16, B=32:
  attention = (32, 8, 1200, 1200) × 2 bytes ≈ 0.88 GB.
All 12 layers are trainable (LoRA on out_proj, linear1, linear2).
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl

os.environ.setdefault("HF_HOME", "/data/rbg/users/ashyamal/huggingface_cache")

REPO_ROOT = Path(__file__).resolve().parents[2]

# peft and transformers are only available in rna_modeling / scgpt_env.
# Import them lazily inside the model classes that need them so this file
# can be imported from any environment (compass, scgpt_env, rna_modeling).

from pytorch_modules.losses.elastic_loss import elastic_loss
from pytorch_modules.losses.cindex_loss import CIndexLoss
from pytorch_modules.losses.cox_loss import cox_ph_loss


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FMFinetuneSurvivalModel(pl.LightningModule):
    """Abstract base for FM fine-tuning survival models.

    Subclasses must implement:
      - _load_and_wrap_fm(): load the backbone, apply LoRA, assign to self.fm
      - _forward_fm(batch) -> Tensor of shape (B, embed_dim)
            Run the FM backbone and return a pooled embedding per sample.
    """

    def __init__(
        self,
        hf_model_id: str,
        embed_dim: int,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        loss_name: str = "CIndex",
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        penalty_weight: float = 0.0,
        use_gradient_checkpointing: bool = False,
        ft_mode: str = "FFT",  # fine-tuning mode: used by COMPASS; ignored by LoRA models
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hf_model_id = hf_model_id
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = list(target_modules) if target_modules else []
        self.embed_dim = embed_dim
        self.loss_name = loss_name
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.penalty_weight = float(penalty_weight)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.ft_mode = ft_mode

        self._load_and_wrap_fm()

        self.head = nn.Linear(embed_dim, 1, bias=True)

        if loss_name == "Cox":
            self.loss_fn = cox_ph_loss
        elif loss_name == "CIndex":
            self.loss_fn = CIndexLoss()
        else:
            raise ValueError(f"Unknown loss_name: {loss_name}. Use 'Cox' or 'CIndex'.")

    def _load_and_wrap_fm(self):
        raise NotImplementedError

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Run the FM backbone; return pooled embedding (B, embed_dim)."""
        raise NotImplementedError

    def forward(self, batch):
        emb = self._forward_fm(batch)           # (B, embed_dim)
        risk = self.head(emb).squeeze(-1)        # (B,)
        return risk

    def training_step(self, batch, batch_idx):
        risk = self(batch)
        pred_loss = self.loss_fn(risk, batch["time"], batch["event"])

        # Skip degenerate batches (all same event status → CIndex undefined → NaN loss).
        # Returning None tells Lightning to skip this micro-step cleanly rather than
        # accumulating a NaN gradient that could corrupt the optimizer state.
        if not torch.isfinite(pred_loss):
            return None

        if self.penalty_weight > 0:
            penalty_l1, penalty_l2 = elastic_loss(self.head)
            loss = pred_loss + self.penalty_weight * (penalty_l1 + penalty_l2)
        else:
            loss = pred_loss

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        risk = self(batch)
        pred_loss = self.loss_fn(risk, batch["time"], batch["event"])
        # Same guard as training_step: skip degenerate batches so a single
        # all-censored or all-event val batch doesn't poison the epoch average.
        if not torch.isfinite(pred_loss):
            return
        self.log("val/loss", pred_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", pred_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        id_ = batch.get("id_", None)
        time = batch["time"].detach().cpu().numpy().tolist()
        event = batch["event"].detach().cpu().numpy().tolist()
        organ = batch.get("organ", None)
        cancer_type = batch.get("cancer_type", None)
        risk = self(batch).detach().float().cpu().numpy().tolist()
        return id_, risk, time, event, organ, cancer_type

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    @property
    def gradient_norm(self) -> torch.Tensor:
        grads = [
            p.grad.norm(p=2) ** 2
            for p in self.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not grads:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(grads).sum().sqrt()

    def on_after_backward(self):
        self.log("train/grad_norm", self.gradient_norm, prog_bar=False)


# ---------------------------------------------------------------------------
# BulkRNABert
# ---------------------------------------------------------------------------

class BulkRNABertFinetuneSurvivalModel(FMFinetuneSurvivalModel):
    """Last-layer LoRA fine-tuning of BulkRNABert for survival.

    Architecture
    ------------
    input_ids (B, 19062)
        → layers 0..N-2: frozen BulkRNABert (no gradient, attention freed immediately)
        → layer N-1: LoRA on w_q, w_k, w_v (gradient computed here only)
        → mean pool over gene dimension → (B, 256)
        → Linear head → risk score (B,)

    Memory rationale
    ----------------
    At B=1, one attention matrix = (1, 8, 19062, 19062) × 2 bytes (bf16) ≈ 5.8 GB.
    Its fp32 gradient ≈ 11.6 GB.  With 4 layers all requiring backward, peak GPU
    memory exceeds 47 GB.  By running layers 0-2 under torch.no_grad(), only the
    last layer's attention gradient (≈ 11.6 GB) is ever live — total peak ≈ 25 GB.
    """

    def _load_and_wrap_fm(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(self.hf_model_id, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            self.hf_model_id,
            config=config,
            trust_remote_code=True,
        )

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.fm = get_peft_model(base_model, lora_cfg)

        # Freeze LoRA params in all but the last transformer layer so that
        # gradients only need to be computed for one layer's attention.
        n_layers = len(base_model.transformer_layers)
        frozen = 0
        for name, param in self.fm.named_parameters():
            if not param.requires_grad:
                continue
            is_last_layer = f"transformer_layers.{n_layers - 1}." in name
            if not is_last_layer:
                param.requires_grad_(False)
                frozen += param.numel()

        self.fm.print_trainable_parameters()
        print(f"  ({frozen:,} additional LoRA params frozen in layers 0–{n_layers - 2})")

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Forward pass: frozen layers 0..N-2, then trained last layer.

        Returns
        -------
        pooled : Tensor, shape (B, embed_dim)
        """
        input_ids = batch["input_ids"]
        base = self.fm.base_model.model

        # Expression embedding + gene positional embedding
        x = base.expression_embedding_layer(input_ids)  # (B, n_genes, embed_dim)
        if base.config.use_gene_embedding:
            gene_ids = torch.arange(base.config.n_genes, device=x.device)
            gene_emb = base.gene_embedding_layer(gene_ids)
            if base.config.project_gene_embedding:
                gene_emb = base.fc_gene_embedding(gene_emb)
            x = x + gene_emb

        B, S = input_ids.shape
        attn_mask = torch.ones((B, 1, S, S), device=x.device, dtype=torch.bool)

        # Layers 0..N-2: no gradient — attention tensors are allocated then
        # immediately freed after each layer; nothing is stored for backward.
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for layer in base.transformer_layers[:-1]:
                    x = layer(x, attention_mask=attn_mask)["embeddings"]

        # Detach so the autograd graph starts fresh at the last layer's input.
        x = x.detach()

        # Last layer: LoRA params are active; attention gradient computed once.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = base.transformer_layers[-1](x, attention_mask=attn_mask)["embeddings"]

        return x.mean(dim=1).float()  # bf16 → float32


# ---------------------------------------------------------------------------
# scGPT
# ---------------------------------------------------------------------------

class scGPTFinetuneSurvivalModel(FMFinetuneSurvivalModel):
    """Full LoRA fine-tuning of scGPT for survival.

    Architecture
    ------------
    gene_ids (B, max_len), values (B, max_len), padding_mask (B, max_len)
        → scGPT TransformerEncoder (12 layers, 512-dim)
            LoRA on out_proj, linear1, linear2 in every layer
        → <cls> token embedding at position 0 → (B, 512)
        → Linear head → risk score (B,)

    The backbone is loaded from a local model directory (not HuggingFace).
    ``hf_model_id`` is repurposed as the path to that directory.

    Memory
    ------
    At B=32, seq_len=1200, 8 heads: attention ≈ 0.88 GB bf16 per layer.
    All 12 layers are trainable; total activation memory is well within 48 GB.
    """

    def _load_and_wrap_fm(self):
        # lazy imports — only available in scgpt_env
        import torchtext
        torchtext.disable_torchtext_deprecation_warning()
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        model_dir = self.hf_model_id  # path to local scGPT model dir
        vocab_file = os.path.join(model_dir, "vocab.json")
        model_file = os.path.join(model_dir, "best_model.pt")
        config_file = os.path.join(model_dir, "args.json")

        with open(config_file) as f:
            cfg = json.load(f)

        vocab = GeneVocab.from_file(vocab_file)
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        self._vocab = vocab

        base_model = TransformerModel(
            ntoken=len(vocab),
            d_model=cfg["embsize"],
            nhead=cfg["nheads"],
            d_hid=cfg["d_hid"],
            nlayers=cfg["nlayers"],
            nlayers_cls=cfg["n_layers_cls"],
            n_cls=1,
            vocab=vocab,
            dropout=cfg["dropout"],
            pad_token=cfg["pad_token"],
            pad_value=cfg["pad_value"],
            do_mvc=cfg.get("MVC", False),
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,  # must be False for LoRA (no flash-attn hooks)
            pre_norm=False,
        )
        load_pretrained(
            base_model,
            torch.load(model_file, map_location="cpu"),
            verbose=False,
        )

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.fm = get_peft_model(base_model, lora_cfg)
        self.fm.print_trainable_parameters()

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Run scGPT encoder; return <cls> embedding (B, embed_dim).

        Uses _encode + _get_cell_emb_from_layer directly to avoid running the
        MLM decoder (not needed for fine-tuning).
        """
        gene_ids = batch["gene_ids"]           # (B, L) int64
        values = batch["values"]               # (B, L) float32, binned
        padding_mask = batch["padding_mask"]   # (B, L) bool, True=pad

        base = self.fm.base_model.model

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # _encode goes through the LoRA-wrapped transformer layers
            output = base._encode(gene_ids, values, padding_mask)          # (B, L, 512)
            cell_emb = base._get_cell_emb_from_layer(output, values)       # (B, 512)

        return cell_emb.float()  # bf16 → float32 (numpy/head don't support bf16)


# ---------------------------------------------------------------------------
# BulkFormer
# ---------------------------------------------------------------------------

class BulkFormerFinetuneSurvivalModel(FMFinetuneSurvivalModel):
    """LoRA fine-tuning of BulkFormer for survival prediction.

    Architecture
    ------------
    input_ids (B, 20010) float32 — log1p expression aligned to BulkFormer gene list
        → PositionalExprEmbedding + ESM2 gene embedding projection + AE encoding
        → 3 × GBFormer (GCNConv + 4 × Performer, linear attention O(N·d))
        → mean-pool over gene dimension → (B, 640)
        → Linear head → risk score (B,)

    LoRA targets the Performer attention projections (to_q, to_k, to_v, to_out)
    in all 3 GBFormer layers.  GCNConv is left frozen (GTEx graph is fixed).

    Memory
    ------
    BulkFormer uses Performer (linear attention), not standard O(N²) attention.
    At B=8, 20010 genes, dim=640: activations ≈ 8 × 20010 × 640 × 4 bytes ≈ 390 MB
    per layer — far more tractable than BulkRNABert.

    ``hf_model_id`` is unused; checkpoint and graph paths are read from the
    BulkFormer repo directory specified in BULKFORMER_DIR.
    """

    BULKFORMER_DIR = os.environ.get("BULKFORMER_DIR", "/data/rbg/users/ashyamal/BulkFormer/BulkFormer")

    def _load_and_wrap_fm(self):
        import sys
        from collections import OrderedDict
        from peft import LoraConfig, get_peft_model

        sys.path.insert(0, self.BULKFORMER_DIR)
        from torch_geometric.typing import SparseTensor
        from utils.BulkFormer import BulkFormer
        from model.config import model_params

        d = self.BULKFORMER_DIR
        graph_idx = torch.load(f"{d}/data/G_gtex.pt", map_location="cpu", weights_only=False)
        graph_wts = torch.load(f"{d}/data/G_gtex_weight.pt", map_location="cpu", weights_only=False)
        graph_cpu = SparseTensor(row=graph_idx[1], col=graph_idx[0], value=graph_wts).t()
        gene_emb = torch.load(f"{d}/data/esm2_feature_concat.pt", map_location="cpu", weights_only=False)

        params = dict(model_params)
        params["graph"] = graph_cpu
        params["gene_emb"] = gene_emb
        base_model = BulkFormer(**params)

        ckpt = torch.load(f"{d}/model/Bulkformer_ckpt_epoch_29.pt", map_location="cpu", weights_only=False)
        new_state = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in ckpt.items()
        )
        base_model.load_state_dict(new_state)

        # Keep a CPU copy of the graph for device management in _forward_fm
        # (SparseTensor is a plain attribute, not an nn.Buffer, so .to(device) is manual)
        self._bulkformer_graph = graph_cpu
        self._graph_device = None

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,  # ["to_q", "to_k", "to_v", "to_out"]
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.fm = get_peft_model(base_model, lora_cfg)
        self.fm.print_trainable_parameters()

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Run BulkFormer encoder; return mean-pooled gene embeddings (B, 640)."""
        x = batch["input_ids"].float()  # ensure float32 — torch_sparse spmm doesn't support bf16

        base = self.fm.base_model.model  # unwrap LoraModel → BulkFormer

        # SparseTensor is a plain attribute — move to input device once, then cache
        dev = str(x.device)
        if self._graph_device != dev:
            base.graph = self._bulkformer_graph.to(x.device)
            self._graph_device = dev

        # Replicate BulkFormer.forward manually so we can apply gradient
        # checkpointing per GBFormer layer. The LoRA modules inside gb_formers
        # are already patched in-place by PEFT, so calling layer(...) uses them.
        x = base.expr_emb(x) + base.gene_emb_proj(base.gene_emb) + base.ae_enc(x).unsqueeze(1)
        x = base.x_proj(x)

        graph = base.graph
        for layer in base.gb_formers:
            if self.use_gradient_checkpointing:
                # use_reentrant=False: closure captures non-Tensor `graph`
                def _run(inp, _layer=layer, _graph=graph):
                    return _layer(inp, _graph)
                x = torch.utils.checkpoint.checkpoint(_run, x, use_reentrant=False)
            else:
                x = layer(x, graph)

        x = base.layernorm(x)
        return x.mean(dim=1).float()  # (B, 640)


# ---------------------------------------------------------------------------
# COMPASS
# ---------------------------------------------------------------------------

class COMPASSFinetuneSurvivalModel(FMFinetuneSurvivalModel):
    """Fine-tuning of COMPASS for survival prediction.

    Architecture
    ------------
    input_ids (B, 1 + n_compass_genes) float32
        col 0 = cancer_code (int stored as float, cast to long by encoder)
        cols 1+ = scaler-transformed log2 gene expression in [0, 1]
        → TransformerEncoder (1-layer Performer, 32-dim)
        → DisentangledProjector → cellpathway embedding (B, 44)
        → Linear head → risk score (B,)

    Fine-tuning modes (set via ft_mode in configs/finetune.yaml)
    ------------------------------------------------------------
    FFT : train all parameters (default — model is tiny, ~100k params)
    PFT : freeze inputencoder, train latentprojector + head
    LFT : freeze inputencoder + latentprojector, train head only

    The COMPASS backbone is loaded from a local pretrainer.pt file.
    ``hf_model_id`` is repurposed as the path to that file.
    LoRA is not used (model is too small to benefit).
    """

    def _load_and_wrap_fm(self):
        import sys
        sys.path.insert(0, os.environ.get("COMPASS_DIR", "/data/rbg/users/ashyamal/COMPASS"))
        from compass import loadcompass

        pretrainer = loadcompass(self.hf_model_id)
        model = pretrainer.model

        if self.ft_mode == "FFT":
            # all parameters trainable — default for COMPASS
            pass
        elif self.ft_mode == "PFT":
            for param in model.inputencoder.parameters():
                param.requires_grad = False
        elif self.ft_mode == "LFT":
            for param in model.inputencoder.parameters():
                param.requires_grad = False
            for param in model.latentprojector.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown ft_mode='{self.ft_mode}'. Use 'FFT', 'PFT', or 'LFT'.")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"COMPASS ft_mode={self.ft_mode}: {trainable:,} / {total:,} params trainable")

        self.fm = model

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Forward pass through COMPASS encoder + projector.

        Returns
        -------
        cellpathway_proj : Tensor, shape (B, 44)
        """
        x = batch["input_ids"]  # (B, 1 + n_compass_genes) float32

        encoding = self.fm.inputencoder(x)                           # (B, L+2, 32)
        _, cellpathway_proj = self.fm.latentprojector(encoding)      # (B, 44)

        return cellpathway_proj


# ---------------------------------------------------------------------------
# CancerFoundation
# ---------------------------------------------------------------------------

class CancerFoundationFinetuneSurvivalModel(FMFinetuneSurvivalModel):
    """LoRA fine-tuning of CancerFoundation (CancerGPT) for survival prediction.

    Architecture
    ------------
    gene_ids (B, max_len), values (B, max_len), padding_mask (B, max_len)
        → CancerGPT TransformerEncoder (6 layers, 256-dim, 8 heads)
            LoRA on out_proj, linear1, linear2 in every TransformerEncoderLayer
        → <cls> token embedding at position 0 → (B, 256)
        → Linear head → risk score (B,)

    Token format matches scGPT: {"gene_ids": int64, "expressions": float32}.
    All patients share the same HVG gene_ids; use cancerfoundation_collate_fn.

    ``hf_model_id`` is unused — weights are loaded from CANCER_FOUNDATION_DIR.
    """

    CANCER_FOUNDATION_DIR = os.environ.get("CANCERFOUNDATION_DIR", "/data/rbg/users/ashyamal/CancerFoundation")

    def _load_and_wrap_fm(self):
        import sys
        sys.path.insert(0, self.CANCER_FOUNDATION_DIR)
        from model.cancergpt import CancerGPT
        from model.vocab import GeneVocab
        from model.utils import load_pretrained
        from peft import LoraConfig, get_peft_model

        model_dir = os.path.join(self.CANCER_FOUNDATION_DIR, "model/assets")
        vocab_file = os.path.join(model_dir, "vocab.json")
        model_file = os.path.join(model_dir, "model.pth")
        config_file = os.path.join(model_dir, "args.json")

        with open(config_file) as f:
            cfg = json.load(f)

        vocab = GeneVocab.from_file(vocab_file)
        vocab.set_default_index(vocab["<pad>"])
        self._pad_token_id = vocab["<pad>"]

        base_model = CancerGPT(
            ntoken=len(vocab),
            d_model=cfg["embsize"],
            nhead=cfg["nheads"],
            d_hid=cfg["d_hid"],
            nlayers=cfg["nlayers"],
            vocab=vocab,
            dropout=cfg["dropout"],
            pad_token="<pad>",
        )
        load_pretrained(
            base_model,
            torch.load(model_file, map_location="cpu"),
            verbose=False,
        )

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.fm = get_peft_model(base_model, lora_cfg)
        self.fm.print_trainable_parameters()

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Run CancerGPT encoder; return <cls> embedding (B, embed_dim)."""
        gene_ids = batch["gene_ids"]          # (B, L) int64
        values = batch["values"]              # (B, L) float32, binned
        padding_mask = batch["padding_mask"]  # (B, L) bool, True=pad

        base = self.fm.base_model.model

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = base.encode(
                gene_ids, values, src_key_padding_mask=padding_mask
            )                                 # (B, L, embed_dim)
            cell_emb = output[:, 0, :]        # <cls> token at position 0

        return cell_emb.float()               # bf16 → float32


# ---------------------------------------------------------------------------
# STATE-SE
# ---------------------------------------------------------------------------

class STATEFinetuneSurvivalModel(FMFinetuneSurvivalModel):
    """LoRA fine-tuning of STATE-SE (SE-600M) for survival prediction.

    Architecture
    ------------
    cell_sentence (B, 2048) int64 — global protein-embedding gene indices
    cell_sentence_counts (B, 2048) float32 — expression weights × 100
        → pe_embedding (Embedding, 145k × 5120, frozen) lookup → (B, 2048, 5120)
        → L2-normalize per gene embedding
        → override position 0 with learnable cls_token
        → encoder: Linear(5120→2048) + LayerNorm + SiLU
        → soft-bin count embeddings added
        → 16 × FlashTransformerEncoderLayer (d_model=2048, nhead=16)
            LoRA on qkv_proj, out_proj, linear1, linear2
        → decoder: SkipBlock(2048) + Linear(2048→2048)
        → CLS at position 0, L2-normalized → (B, 2048)
        → Linear survival head → risk score (B,)

    Memory
    ------
    FlashTransformerEncoderLayer uses scaled_dot_product_attention (Flash SDPA)
    which does not materialise the full (B, nhead, 2048, 2048) attention matrix.
    At bf16 with B=8: activations ≈ 8 × 2048 × 2048 × 2 bytes ≈ 67 MB per layer
    (dominant), manageable on a 48 GB GPU.
    Use gradient_checkpointing: true in the config to trade compute for memory.

    ``hf_model_id`` is unused — checkpoint is loaded from STATE_MODEL_DIR.
    """

    STATE_REPO_DIR = os.environ.get("STATE_REPO_DIR", "/data/rbg/users/ashyamal/state/src")
    STATE_MODEL_DIR = os.environ.get("STATE_MODEL_DIR", str(REPO_ROOT / "data" / "STATE" / "SE-600M"))

    def __init__(self, *args, lora_n_layers: int = -1, **kwargs):
        # Must be set before super().__init__() because _load_and_wrap_fm() is called there.
        self.lora_n_layers = lora_n_layers
        super().__init__(*args, **kwargs)

    def _load_and_wrap_fm(self):
        import sys
        sys.path.insert(0, self.STATE_REPO_DIR)
        from peft import LoraConfig, get_peft_model
        from omegaconf import OmegaConf
        from state.emb.nn.model import StateEmbeddingModel

        ckpt_path = os.path.join(self.STATE_MODEL_DIR, "se600m_epoch16.ckpt")
        cfg_path = os.path.join(self.STATE_MODEL_DIR, "config.yaml")
        pe_path = os.path.join(self.STATE_MODEL_DIR, "protein_embeddings.pt")

        cfg = OmegaConf.load(cfg_path)

        # Load backbone from checkpoint
        base_model = StateEmbeddingModel.load_from_checkpoint(
            ckpt_path,
            dropout=0.0,
            strict=False,
            cfg=cfg,
            weights_only=False,
        )

        # Set up frozen protein-embedding lookup (ESM2, 145k × 5120)
        protein_embeds = torch.load(pe_path, map_location="cpu", weights_only=False)
        all_pe = torch.vstack(list(protein_embeds.values()))
        base_model.pe_embedding = nn.Embedding.from_pretrained(all_pe, freeze=True)

        # We don't need the MLM gene-prediction decoder during survival fine-tuning;
        # freeze it to avoid computing unnecessary gradients.
        for p in base_model.binary_decoder.parameters():
            p.requires_grad_(False)

        # Apply LoRA to the transformer layers
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,  # qkv_proj, out_proj, linear1, linear2
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.fm = get_peft_model(base_model, lora_cfg)

        # Optionally restrict LoRA to the last N transformer layers.
        # lora_n_layers=-1 (default) trains all layers.
        # e.g. lora_n_layers=4 freezes LoRA in layers 0-11, only trains layers 12-15.
        # Read from hparams if set via set_lora_n_layers() before construction.
        n_layers = len(base_model.transformer_encoder.layers)
        lora_n_layers = getattr(self, "lora_n_layers", -1)
        if lora_n_layers > 0:
            frozen = 0
            for name, param in self.fm.named_parameters():
                if not param.requires_grad:
                    continue
                # check if this param belongs to one of the last lora_n_layers layers
                is_active = any(
                    f"transformer_encoder.layers.{n_layers - k}." in name
                    for k in range(1, lora_n_layers + 1)
                )
                if not is_active:
                    param.requires_grad_(False)
                    frozen += param.numel()
            print(f"  (frozen {frozen:,} LoRA params in layers 0–{n_layers - lora_n_layers - 1})")

        self.fm.print_trainable_parameters()

    def _forward_fm(self, batch: dict) -> torch.Tensor:
        """Run STATE-SE encoder; return L2-normalized CLS embedding (B, 2048).

        Replicates _compute_embedding_for_batch → forward, but with gradients
        enabled for the LoRA-wrapped transformer layers.
        """
        import math
        import torch.nn.functional as F

        cell_sentence = batch["cell_sentence"].to(self.device)         # (B, 2048) int64
        cell_sentence_counts = batch["cell_sentence_counts"].to(self.device)  # (B, 2048) float32

        base = self.fm.base_model.model  # StateEmbeddingModel (LoRA-patched in-place)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 1. Lookup ESM2 protein embeddings → (B, 2048, 5120)
            src = base.pe_embedding(cell_sentence)

            # 2. L2-normalise each gene token (mirrors _compute_embedding_for_batch)
            src = F.normalize(src, dim=2)

            # 3. Override position 0 with the learnable CLS token
            src = src.clone()
            src[:, 0, :] = base.cls_token.expand(src.size(0), -1)

            # 4. Append learnable dataset token if model uses dataset correction
            counts_in = cell_sentence_counts
            if base.dataset_token is not None:
                ds_tok = base.dataset_token.expand(src.size(0), -1).unsqueeze(1)  # (B,1,5120)
                src = torch.cat([src, ds_tok], dim=1)                             # (B,2049,5120)
                ds_pad = torch.zeros(counts_in.size(0), 1, device=self.device, dtype=counts_in.dtype)
                counts_in = torch.cat([counts_in, ds_pad], dim=1)                 # (B,2049)

            # 5. Project: Linear(5120→2048) + LayerNorm + SiLU, scaled by sqrt(d_model)
            hidden = base.encoder(src) * math.sqrt(base.d_model)  # (B, seq, 2048)

            # 6. Add soft-binned count embeddings
            counts_3d = counts_in.unsqueeze(-1)                           # (B, seq, 1)
            bin_weights = F.softmax(base.count_encoder(counts_3d), dim=-1)  # (B, seq, 10)
            bin_indices = torch.arange(10, device=self.device)
            bin_emb = base.bin_encoder(bin_indices)                       # (10, 2048)
            hidden = hidden + torch.matmul(bin_weights, bin_emb)          # (B, seq, 2048)

            # 7. Run through 16 FlashTransformer layers (LoRA active here)
            if self.use_gradient_checkpointing:
                output = hidden
                for layer in base.transformer_encoder.layers:
                    def _run(inp, _layer=layer):
                        return _layer(inp)
                    output = torch.utils.checkpoint.checkpoint(_run, output, use_reentrant=False)
            else:
                output = base.transformer_encoder(hidden)  # (B, seq, 2048)

            # 8. Decode and extract CLS token
            gene_output = base.decoder(output)          # (B, seq, 2048)
            embedding = gene_output[:, 0, :]            # CLS at position 0 → (B, 2048)
            embedding = F.normalize(embedding, dim=1)   # L2-normalise (matches pre-training)

        return embedding.float()   # bf16 → float32


# ---------------------------------------------------------------------------
# Registry (add future FMs here)
# ---------------------------------------------------------------------------

FM_REGISTRY: dict = {
    "BulkRNABert": BulkRNABertFinetuneSurvivalModel,
    "scGPT": scGPTFinetuneSurvivalModel,
    "COMPASS": COMPASSFinetuneSurvivalModel,
    "BulkFormer": BulkFormerFinetuneSurvivalModel,
    "CancerFoundation": CancerFoundationFinetuneSurvivalModel,
    "STATE": STATEFinetuneSurvivalModel,
}

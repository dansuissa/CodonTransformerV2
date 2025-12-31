import argparse
import os
from typing import Optional
import lightning.pytorch as pl
import torch
import torch.nn as nn
import webdataset as wds
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast
from codontransformer2.dataset import MaskedTokenizerCollator
from codontransformer2.dataset.constants import TOKEN2MASK
from codontransformer2.eval.CT2_eval import (
    build_species_buckets,
    evaluate_mlm_batch,
    evaluate_species_ablations,
    evaluate_tail_buckets,
)


class TrainHarness(pl.LightningModule):
    def __init__(
        self,
        model: ModernBertForMaskedLM,
        n_species: int,
        learning_rate: float,
        warmup_fraction: float,
        learning_rate_decay: float,
        weight_decay: float,
        unknown_species_id: int,
        eval_topk: int = 5,
        synonym_topk: int = 1,
        do_species_ablations_every: int = 50,
        head_frac: float = 0.10,
        tail_frac: float = 0.50,
        species_counts_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        # Species embedding table: [S, H]
        self.species_embed = nn.Embedding(n_species, model.config.hidden_size)

        # Build codon_id_to_aa_id (AA-class IDs) using TOKEN2MASK mapping.
        # Works for "synonymous accuracy" because codons mapping to same amino acid share same AA ID.
        vocab_size = model.config.vocab_size
        codon2aa = torch.full((vocab_size,), -1, dtype=torch.long)
        for codon_id, aa_id in TOKEN2MASK.items():
            if 0 <= codon_id < vocab_size:
                codon2aa[codon_id] = aa_id
        self.register_buffer("codon_id_to_aa_id", codon2aa)

        # Optional: species frequency counts from TRAIN data for head/mid/tail buckets
        self.buckets = None
        if species_counts_path is not None:
            counts = torch.load(species_counts_path, map_location="cpu")
            if (not torch.is_tensor(counts)) or counts.dim() != 1:
                raise ValueError(
                    "species_counts_path must be a torch-saved 1D Tensor [S]. "
                    f"Got type={type(counts)} shape={getattr(counts, 'shape', None)}"
                )
            self.register_buffer("species_counts", counts.long())

    def _get_lr(self):
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        return opt.param_groups[0].get("lr", None)

    def _forward_with_species(self, batch, species_ids: torch.Tensor):
        # token_embeds: [B,L,H]
        token_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        sp_vec = self.species_embed(species_ids)  # [B,H]
        token_embeds = token_embeds + sp_vec[:, None, :]  # [B,L,H]
        out = self.model(
            inputs_embeds=token_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return out

    # ---------------------------
    # TRAIN
    # ---------------------------
    def training_step(self, batch, batch_idx):
        out = self._forward_with_species(batch, batch["species_id"])
        self.log_dict(
            {"loss": out.loss, "lr": self._get_lr()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return out.loss

    # ---------------------------
    # TEST (your "evaluation set")
    # ---------------------------
    def on_test_start(self):
        # Build head/mid/tail buckets once (if counts provided)
        if hasattr(self, "species_counts") and self.buckets is None:
            self.buckets = build_species_buckets(
                self.species_counts.to(self.device),
                head_frac=self.hparams.head_frac,
                tail_frac=self.hparams.tail_frac,
            )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # Core MLM metrics
        metrics = evaluate_mlm_batch(
            model=self.model,
            species_embed=self.species_embed,
            batch=batch,
            topk=self.hparams.eval_topk,
            codon_id_to_aa_id=self.codon_id_to_aa_id,
            synonym_topk=self.hparams.synonym_topk,
        )

        # Make test keys explicit (Lightning will aggregate them)
        metrics = {("test_" + k if not k.startswith("val_") else k.replace("val_", "test_")): v
                   for k, v in metrics.items()}

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Species shortcut diagnostics every N batches (3 forward passes)
        if self.hparams.do_species_ablations_every > 0 and (batch_idx % self.hparams.do_species_ablations_every == 0):
            ab = evaluate_species_ablations(
                model=self.model,
                species_embed=self.species_embed,
                batch=batch,
                unknown_species_id=self.hparams.unknown_species_id,
                topk=self.hparams.eval_topk,
            )
            ab = {k.replace("val_", "test_"): v for k, v in ab.items()}
            self.log_dict(ab, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            # Head/mid/tail metrics (optional)
            if self.buckets is not None:
                tb = evaluate_tail_buckets(
                    model=self.model,
                    species_embed=self.species_embed,
                    batch=batch,
                    buckets=self.buckets,
                    topk=self.hparams.eval_topk,
                )
                tb = {k.replace("val_", "test_"): v for k, v in tb.items()}
                self.log_dict(tb, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    # ---------------------------
    # OPTIM
    # ---------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_fraction * total_steps)
        decay_steps = max(1, total_steps - warmup_steps)

        print(f"Optimizer configured with LR: {self.hparams.learning_rate}")
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Decay steps: {decay_steps}")

        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        decay = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.hparams.learning_rate * self.hparams.learning_rate_decay,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class EpochCheckpoint(pl.Callback):
    def __init__(self, save_interval: int, checkpoint_dir: str):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if self.save_interval > 0 and (current_epoch % self.save_interval == 0 or current_epoch == 0):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}\n")


def make_wds_loader(
    shard_pattern: str,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    num_workers: int,
    shardshuffle: int,
    pin_memory: bool,
    limit_samples: Optional[int] = None,
):
    ds = wds.WebDataset(
        shard_pattern,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
        shardshuffle=(False if shardshuffle <= 0 else shardshuffle),
    )

    # Optional: limit samples for quick smoke test
    if limit_samples is not None and limit_samples > 0:
        ds = ds.with_length(limit_samples)

    loader = DataLoader(
        dataset=ds,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=pin_memory,
    )
    return loader


def main(args):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Enable CUDA knobs only if CUDA exists (keeps MPS safe)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        max_len=args.max_length,
    )

    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        max_position_embeddings=args.max_length,
        attn_implementation=args.attn_implementation,
    )
    model = ModernBertForMaskedLM(config=config)

    n_species = args.num_organisms + args.extra_organisms

    # unknown_species_id default: last slot
    unknown_species_id = args.unknown_species_id
    if unknown_species_id < 0:
        unknown_species_id = n_species - 1
    if not (0 <= unknown_species_id < n_species):
        raise ValueError(f"unknown_species_id must be in [0, {n_species-1}], got {unknown_species_id}")

    harnessed_model = TrainHarness(
        model=model,
        n_species=n_species,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        learning_rate_decay=args.learning_rate_decay,
        weight_decay=args.weight_decay,
        unknown_species_id=unknown_species_id,
        eval_topk=args.eval_topk,
        synonym_topk=args.synonym_topk,
        do_species_ablations_every=args.do_species_ablations_every,
        head_frac=args.head_frac,
        tail_frac=args.tail_frac,
        species_counts_path=args.species_counts_path,
    )

    # pin_memory only meaningful for CUDA
    pin_memory = bool(args.pin_memory and torch.cuda.is_available())

    # -------------------
    # TRAIN loader
    # -------------------
    train_pattern = os.path.join(args.dataroot, args.shard_pattern)
    train_loader = make_wds_loader(
        shard_pattern=train_pattern,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shardshuffle=args.shardshuffle,
        pin_memory=pin_memory,
    )

    # -------------------
    # TEST loader (used for evaluation; no validation loop)
    # -------------------
    test_loader = None
    if args.test_dataroot is not None and args.test_shard_pattern is not None:
        test_pattern = os.path.join(args.test_dataroot, args.test_shard_pattern)
        test_loader = make_wds_loader(
            shard_pattern=test_pattern,
            tokenizer=tokenizer,
            batch_size=(args.test_batch_size or args.batch_size),
            num_workers=args.num_workers,
            shardshuffle=0,  # keep deterministic ordering
            pin_memory=pin_memory,
        )

    callbacks = []
    if args.save_interval > 0:
        callbacks.append(EpochCheckpoint(args.save_interval, args.checkpoint_dir))

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy=args.strategy,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        limit_train_batches=args.limit_train_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    trainer.fit(harnessed_model, train_loader)

    # Run evaluations on TEST data (optional but recommended)
    if test_loader is not None:
        trainer.test(harnessed_model, dataloaders=test_loader, ckpt_path=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CodonTransformer2 and evaluate on a test set (no validation).")

    # Data (train)
    parser.add_argument("--dataroot", type=str, required=True, help="Root directory for training data")
    parser.add_argument(
        "--shard_pattern",
        type=str,
        default="shard-{000000..003863}.tar",
        help="Pattern for training shards (supports .tar or .tar.gz if your files are .tar.gz)",
    )
    parser.add_argument("--tokenizer_file", type=str, required=True, help="Path to tokenizer json")
    parser.add_argument("--shardshuffle", type=int, default=0, help="0 disables shard shuffle; set e.g. 1000")

    # Data (test / evaluation)
    parser.add_argument("--test_dataroot", type=str, default=None, help="Root directory for test shards")
    parser.add_argument("--test_shard_pattern", type=str, default=None, help="Pattern for test shards")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Test batch size (defaults to train)")

    # Model
    parser.add_argument("--num_organisms", type=int, default=26678, help="Number of known organisms")
    parser.add_argument("--extra_organisms", type=int, default=2000, help="Extra organism slots")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="flash_attention_2 | sdpa | eager",
    )

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_decay", type=float, default=0.1)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_train_batches", type=int, default=400_000)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--pin_memory", action="store_true", help="Only effective on CUDA")

    # Evaluation knobs (test loop)
    parser.add_argument(
        "--unknown_species_id",
        type=int,
        default=-1,
        help="UNKNOWN species ID. If -1, uses last slot (n_species-1).",
    )
    parser.add_argument(
        "--species_counts_path",
        type=str,
        default=None,
        help="Path to torch-saved tensor [S] with TRAIN counts per species_id (optional, for head/mid/tail).",
    )
    parser.add_argument("--eval_topk", type=int, default=5)
    parser.add_argument("--synonym_topk", type=int, default=1)
    parser.add_argument(
        "--do_species_ablations_every",
        type=int,
        default=50,
        help="Run (correct/unknown/wrong) species ablations every N test batches. 0 disables.",
    )
    parser.add_argument("--head_frac", type=float, default=0.10)
    parser.add_argument("--tail_frac", type=float, default=0.50)

    # Hardware
    parser.add_argument("--strategy", type=str, default="auto", help="deepspeed | ddp | auto (MPS can't do deepspeed)")
    parser.add_argument("--accelerator", type=str, default="auto", help="gpu | cpu | mps | auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default=".")
    parser.add_argument("--save_interval", type=int, default=1)

    # Other
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    main(args)

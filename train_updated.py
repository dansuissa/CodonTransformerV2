import argparse
import os

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
        model,
        n_species,
        learning_rate,
        warmup_fraction,
        learning_rate_decay,
        weight_decay,
        unknown_species_id: int,
        eval_topk: int = 5,
        synonym_topk: int = 1,
        do_species_ablations_every: int = 50,
        head_frac: float = 0.10,
        tail_frac: float = 0.50,
        species_counts_path: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.species_embed = nn.Embedding(n_species, model.config.hidden_size)

        # --- Build codon_id_to_aa_id using TOKEN2MASK (for synonymous accuracy) ---
        vocab_size = model.config.vocab_size
        codon2aa = torch.full((vocab_size,), -1, dtype=torch.long)
        for codon_id, aa_id in TOKEN2MASK.items():
            if 0 <= codon_id < vocab_size:
                codon2aa[codon_id] = aa_id
        self.register_buffer("codon_id_to_aa_id", codon2aa)

        # --- Optional: species frequency counts for head/mid/tail buckets ---
        self.species_counts = None
        self.buckets = None
        if species_counts_path is not None:
            counts = torch.load(species_counts_path, map_location="cpu")
            if (not torch.is_tensor(counts)) or counts.dim() != 1:
                raise ValueError(
                    f"species_counts_path must contain a 1D torch Tensor [S], got: {type(counts)} shape={getattr(counts,'shape',None)}"
                )
            self.register_buffer("species_counts_buf", counts.long())
            self.species_counts = self.species_counts_buf

    def _get_lr(self):
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        return opt.param_groups[0].get("lr", None)

    def training_step(self, batch, batch_idx):
        token_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        sp_vec = self.species_embed(batch["species_id"])  # [B, H]
        token_embeds = token_embeds + sp_vec[:, None, :]  # [B, L, H]

        outputs = self.model(
            inputs_embeds=token_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        self.log_dict(
            {
                "loss": outputs.loss,
                "lr": self._get_lr(),
            },
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return outputs.loss

    def on_validation_start(self):
        # Build head/mid/tail buckets once (if counts provided)
        if self.species_counts is not None and self.buckets is None:
            self.buckets = build_species_buckets(
                self.species_counts.to(self.device),
                head_frac=self.hparams.head_frac,
                tail_frac=self.hparams.tail_frac,
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Core MLM validation metrics
        metrics = evaluate_mlm_batch(
            model=self.model,
            species_embed=self.species_embed,
            batch=batch,
            topk=self.hparams.eval_topk,
            codon_id_to_aa_id=self.codon_id_to_aa_id,
            synonym_topk=self.hparams.synonym_topk,
        )
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Species shortcut diagnostics every N batches
        if batch_idx % self.hparams.do_species_ablations_every == 0:
            ab = evaluate_species_ablations(
                model=self.model,
                species_embed=self.species_embed,
                batch=batch,
                unknown_species_id=self.hparams.unknown_species_id,
                topk=self.hparams.eval_topk,
            )
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
                self.log_dict(tb, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

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
    def __init__(self, save_interval, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_interval == 0 or current_epoch == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}\n")


def main(args):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # enable CUDA knobs only if CUDA exists (keeps MPS safe)
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

    # -------------------
    # TRAIN loader
    # -------------------
    train_pattern = os.path.join(args.dataroot, args.shard_pattern)
    train_data = wds.WebDataset(
        train_pattern,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
        shardshuffle=args.shardshuffle,
    )
    train_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=args.pin_memory,
    )


    val_loader = None
    if args.val_dataroot is not None and args.val_shard_pattern is not None:
        val_pattern = os.path.join(args.val_dataroot, args.val_shard_pattern)
        val_data = wds.WebDataset(
            val_pattern,
            nodesplitter=wds.shardlists.split_by_node,
            workersplitter=wds.shardlists.split_by_worker,
            shardshuffle=False,
        )
        val_loader = DataLoader(
            dataset=val_data,
            collate_fn=MaskedTokenizerCollator(tokenizer),
            batch_size=args.val_batch_size or args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
            pin_memory=args.pin_memory,
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
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    trainer.fit(harnessed_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training harness with evaluation.")
    parser.add_argument("--dataroot", type=str, required=True, help="Root directory for training data")
    parser.add_argument("--shard_pattern", type=str, default="shard-{000000..003863}.tar")
    parser.add_argument("--tokenizer_file", type=str, default="data/codon_transformer_tokenizer.json")
    parser.add_argument("--shardshuffle", type=int, default=0, help="0 disables shard shuffle; set e.g. 1000")

    parser.add_argument("--val_dataroot", type=str, default=None, help="Root directory for val shards")
    parser.add_argument("--val_shard_pattern", type=str, default=None, help="Pattern for val shards")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Val batch size (defaults to train)")
    parser.add_argument("--limit_val_batches", type=int, default=200, help="Limit val batches each val run")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validate every N epochs")

    parser.add_argument("--num_organisms", type=int, default=4742)
    parser.add_argument("--extra_organisms", type=int, default=2000)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="flash_attention_2 | sdpa | eager",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_decay", type=float, default=0.1)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_train_batches", type=int, default=400_000)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--pin_memory", action="store_true")

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
        help="Path to torch-saved tensor [S] with train counts per species_id (optional).",
    )
    parser.add_argument("--eval_topk", type=int, default=5)
    parser.add_argument("--synonym_topk", type=int, default=1)
    parser.add_argument("--do_species_ablations_every", type=int, default=50)
    parser.add_argument("--head_frac", type=float, default=0.10)
    parser.add_argument("--tail_frac", type=float, default=0.50)
    parser.add_argument("--strategy", type=str, default="deepspeed")
    parser.add_argument("--accelerator", type=str, default="gpu", help="gpu | cpu | mps | auto")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--checkpoint_dir", type=str, default=".")
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    main(args)

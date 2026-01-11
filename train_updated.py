from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import lightning.pytorch as pl
import torch
import torch.nn as nn
import webdataset as wds
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast
from codontransformer2.dataset.collators import MaskedTokenizerCollator
from codontransformer2.dataset.constants import TOKEN2MASK
from codontransformer2.eval.CT2_eval import (
    build_species_buckets,
    evaluate_mlm_batch,
    evaluate_species_ablations,
    evaluate_tail_buckets,
)


class OrganismPerJSONGenesDataset(IterableDataset):
    """
    Reads per-organism JSON files like:
      {
        "genome_id": "...",
        "species": "...",
        "test_gene_count": 173,
        "genes": [{"protein_id": "...", "dna_sequence": "...", ...}, ...]
      }

    Yields samples compatible with the collator:
      {"json": b'{"species": "...", "genome_id": "...", "protein_id": "...", "dna_sequence": "..."}'}
    """

    def __init__(self, json_dir: str):
        super().__init__()
        self.json_dir = Path(json_dir)

        if not self.json_dir.exists():
            raise FileNotFoundError(f"Test JSON dir not found: {self.json_dir}")

        self.files = sorted([p for p in self.json_dir.glob("*.json") if p.is_file()])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .json files found in: {self.json_dir}")

    def __iter__(self) -> Iterator[Dict]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        # partition files across workers deterministically
        files = self.files[worker_id::num_workers]

        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                org = json.load(f)

            species = org.get("species", None)
            genome_id = org.get("genome_id", None)
            genes = org.get("genes", [])

            for g in genes:
                dna = g.get("dna_sequence", None)
                if dna is None:
                    continue

                sample = {
                    "species": species,
                    "genome_id": genome_id,
                    "protein_id": g.get("protein_id", None),
                    "dna_sequence": dna,
                    "protein_sequence": g.get("protein_sequence", None),
                }
                yield {"json": json.dumps(sample).encode("utf-8")}


class TrainHarness(pl.LightningModule):
    def __init__(
        self,
        model,
        *,
        n_species_total: int,
        learning_rate: float,
        warmup_fraction: float,
        learning_rate_decay: float,
        weight_decay: float,
        unknown_species_id: int,
        eval_topk: int,
        synonym_topk: int,
        do_species_ablations_every: int,
        head_frac: float,
        tail_frac: float,
        species_counts_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.species_embed = nn.Embedding(n_species_total, model.config.hidden_size)
        vocab_size = model.config.vocab_size
        codon2aa = torch.full((vocab_size,), -1, dtype=torch.long)
        for codon_id, aa_id in TOKEN2MASK.items():
            if 0 <= codon_id < vocab_size:
                codon2aa[codon_id] = int(aa_id)
        self.register_buffer("codon_id_to_aa_id", codon2aa)
        self.buckets = None
        self.species_counts = None
        if species_counts_path is not None:
            counts = torch.load(species_counts_path, map_location="cpu")
            if (not torch.is_tensor(counts)) or counts.dim() != 1:
                raise ValueError("species_counts_path must be a torch-saved 1D tensor [S].")
            self.register_buffer("species_counts_buf", counts.long())
            self.species_counts = self.species_counts_buf

    def _get_lr(self):
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        return opt.param_groups[0].get("lr", None)

    def training_step(self, batch, batch_idx):
        token_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        sp_vec = self.species_embed(batch["species_id"])  # [B,H]
        token_embeds = token_embeds + sp_vec[:, None, :]  # [B,L,H]

        out = self.model(
            inputs_embeds=token_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        self.log_dict(
            {"loss": out.loss, "lr": self._get_lr()},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return out.loss

    def on_validation_start(self):
        # We use "validation" loop as "test-eval during training" (since you said: no val split)
        if self.species_counts is not None and self.buckets is None:
            self.buckets = build_species_buckets(
                self.species_counts.to(self.device),
                head_frac=self.hparams.head_frac,
                tail_frac=self.hparams.tail_frac,
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Core metrics
        m = evaluate_mlm_batch(
            model=self.model,
            species_embed=self.species_embed,
            batch=batch,
            topk=self.hparams.eval_topk,
            codon_id_to_aa_id=self.codon_id_to_aa_id,
            synonym_topk=self.hparams.synonym_topk,
            prefix="test",
        )
        self.log_dict(m, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Species shortcut diagnostics
        if self.hparams.do_species_ablations_every > 0 and (batch_idx % self.hparams.do_species_ablations_every == 0):
            ab = evaluate_species_ablations(
                model=self.model,
                species_embed=self.species_embed,
                batch=batch,
                unknown_species_id=self.hparams.unknown_species_id,
                topk=self.hparams.eval_topk,
                prefix="test",
            )
            self.log_dict(ab, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            if self.buckets is not None:
                tb = evaluate_tail_buckets(
                    model=self.model,
                    species_embed=self.species_embed,
                    batch=batch,
                    buckets=self.buckets,
                    topk=self.hparams.eval_topk,
                    prefix="test",
                )
                self.log_dict(tb, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(self.hparams.warmup_fraction * total_steps)
        decay_steps = max(1, total_steps - warmup_steps)

        print(f"Total steps: {total_steps} | Warmup: {warmup_steps} | Decay: {decay_steps}")

        warmup = LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=max(1, warmup_steps))
        decay = CosineAnnealingLR(opt, T_max=decay_steps, eta_min=self.hparams.learning_rate * self.hparams.learning_rate_decay)
        sched = SequentialLR(opt, schedulers=[warmup, decay], milestones=[warmup_steps])

        return [opt], [{"scheduler": sched, "interval": "step"}]


class EpochCheckpoint(pl.Callback):
    def __init__(self, save_interval: int, checkpoint_dir: str):
        super().__init__()
        self.save_interval = int(save_interval)
        self.checkpoint_dir = checkpoint_dir

    def on_train_epoch_end(self, trainer, pl_module):
        e = trainer.current_epoch
        if self.save_interval <= 0:
            return
        if e % self.save_interval == 0 or e == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, f"epoch_{e}.ckpt")
            trainer.save_checkpoint(path)
            print(f"\nCheckpoint saved: {path}\n")


def load_species_to_id(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # ensure int
    return {str(k): int(v) for k, v in m.items()}


def main(args):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Tokenizer
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

    # Model config
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        max_position_embeddings=args.max_length,
        attn_implementation=args.attn_implementation,
    )
    model = ModernBertForMaskedLM(config=config)

    # Species spaces
    n_real_species = args.num_organisms
    n_species_total = n_real_species + args.extra_organisms

    unknown_species_id = args.unknown_species_id
    if unknown_species_id < 0:
        # Recommended: FIRST empty slot = n_real_species
        unknown_species_id = n_real_species

    if not (0 <= unknown_species_id < n_species_total):
        raise ValueError(f"unknown_species_id must be in [0,{n_species_total-1}], got {unknown_species_id}")

    # Mapping
    species_to_id_path = args.species_to_id_path
    if species_to_id_path is None:
        species_to_id_path = os.path.join(args.dataroot, "species_to_id.json")
    species_to_id = load_species_to_id(species_to_id_path)

    # Collator (same for train + test dataset)
    collator = MaskedTokenizerCollator(
        tokenizer,
        species_to_id=species_to_id,
        unknown_species_id=unknown_species_id,
        max_species_id=n_species_total,
        species_dropout_prob=args.species_dropout_prob,
        mlm_probability=args.mlm_probability,
    )

    # ------------------------
    # TRAIN: WebDataset shards
    # ------------------------
    train_pattern = os.path.join(args.dataroot, args.shard_pattern)
    train_data = wds.WebDataset(
        train_pattern,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
        shardshuffle=args.shardshuffle if args.shardshuffle > 0 else False,
    )

    pin_memory = bool(args.pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        dataset=train_data,
        collate_fn=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=pin_memory,
    )

    # --------------------------------
    # TEST (used as "val" during train)
    # per-organism JSONs (not shards)
    # --------------------------------
    test_loader = None
    if args.test_json_dir is not None:
        test_ds = OrganismPerJSONGenesDataset(args.test_json_dir)
        test_loader = DataLoader(
            dataset=test_ds,
            collate_fn=collator,
            batch_size=args.test_batch_size or args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
            pin_memory=pin_memory,
        )

    # Lightning module
    harness = TrainHarness(
        model=model,
        n_species_total=n_species_total,
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

    callbacks: List[pl.Callback] = []
    callbacks.append(EpochCheckpoint(args.save_interval, args.checkpoint_dir))

    # If you run on MPS, deepspeed won't work; set strategy accordingly.
    strategy = args.strategy
    if args.accelerator == "mps" and strategy == "deepspeed":
        print("NOTE: accelerator=mps does not support deepspeed; switching strategy to 'auto'.")
        strategy = "auto"

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_test_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_test_every_n_epoch,
    )

    # Important: we pass test_loader as "val" loader so you get eval metrics during training
    trainer.fit(harness, train_loader, test_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser("CodonTransformer2 training (test-eval during training)")

    # Data
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--shard_pattern", type=str, default="shard-{000000..003863}.tar.gz")
    p.add_argument("--tokenizer_file", type=str, required=True)
    p.add_argument("--species_to_id_path", type=str, default=None)

    # Test jsons (per-organism folder)
    p.add_argument("--test_json_dir", type=str, default=None, help="e.g. .../test_sets_by_organism")
    p.add_argument("--test_batch_size", type=int, default=None)
    p.add_argument("--limit_test_batches", type=int, default=200)
    p.add_argument("--check_test_every_n_epoch", type=int, default=1)

    # Species space
    p.add_argument("--num_organisms", type=int, default=26678)
    p.add_argument("--extra_organisms", type=int, default=2000)
    p.add_argument("--unknown_species_id", type=int, default=-1, help="-1 => uses first empty slot (num_organisms)")
    p.add_argument("--species_counts_path", type=str, default=None, help="torch file with [S] train counts per species_id")

    # Masking + species dropout
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--species_dropout_prob", type=float, default=0.0)

    # Model/training
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--attn_implementation", type=str, default="sdpa", help="flash_attention_2 | sdpa | eager")

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--learning_rate_decay", type=float, default=0.1)
    p.add_argument("--warmup_fraction", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)

    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--limit_train_batches", type=int, default=400_000)
    p.add_argument("--log_every_n_steps", type=int, default=10)

    # Eval knobs
    p.add_argument("--eval_topk", type=int, default=5)
    p.add_argument("--synonym_topk", type=int, default=1)
    p.add_argument("--do_species_ablations_every", type=int, default=50)
    p.add_argument("--head_frac", type=float, default=0.10)
    p.add_argument("--tail_frac", type=float, default=0.50)

    # Loader shuffle
    p.add_argument("--shardshuffle", type=int, default=0, help="0 disables shard shuffle; set e.g. 1000")

    # Hardware
    p.add_argument("--accelerator", type=str, default="gpu", help="gpu | cpu | mps | auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--strategy", type=str, default="deepspeed", help="deepspeed | ddp | auto")
    p.add_argument("--precision", type=str, default="bf16-mixed")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default=".")
    p.add_argument("--save_interval", type=int, default=1)

    # Other
    p.add_argument("--seed", type=int, default=123)

    args = p.parse_args()
    main(args)

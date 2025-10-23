import argparse
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
)

from codontransformer2.dataset import MaskedTokenizerCollator


class TrainHarness(pl.LightningModule):
    def __init__(self, model, n_species, learning_rate, warmup_fraction):
        super().__init__()
        self.model = model
        self.species_embed = nn.Embedding(n_species, model.config.hidden_size)
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_fraction,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        token_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        sp_vec = self.species_embed(batch["species_id"])  # [B, H]
        sp_vec = sp_vec.unsqueeze(1).expand_as(token_embeds)  # [B, L, H]
        inputs_embeds = token_embeds + sp_vec

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return outputs.loss


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
        attn_implementation="flash_attention_2",
    )
    model = ModernBertForMaskedLM(config=config)
    harnessed_model = TrainHarness(
        model,
        n_species=args.num_organisms + args.extra_organisms,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
    )

    shard_pattern = os.path.join(args.dataroot, args.shard_pattern)
    train_data = wds.WebDataset(
        shard_pattern,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
    )
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    callbacks = []
    if args.save_interval > 0:
        callbacks.append(EpochCheckpoint(args.save_interval, args.checkpoint_dir))

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy=args.strategy,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        limit_train_batches=args.limit_train_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
    )

    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training harness.")

    # Data arguments
    parser.add_argument("--dataroot", type=str, required=True, help="Root directory for data")
    parser.add_argument(
        "--shard-pattern",
        type=str,
        default="shard-{000000..003863}.tar",
        help="Pattern for data shards",
    )
    parser.add_argument(
        "--tokenizer-file",
        type=str,
        default="data/codon_transformer_tokenizer.json",
        help="Path to tokenizer file",
    )

    # Model arguments
    parser.add_argument("--num-organisms", type=int, default=4742, help="Number of organisms")
    parser.add_argument("--extra-organisms", type=int, default=2000, help="Extra organism slots")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-fraction", type=float, default=0.1, help="Warmup fraction")
    parser.add_argument("--max-epochs", type=int, default=5, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument(
        "--limit-train-batches", type=int, default=400_000, help="Limit training batches"
    )
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Log every n steps")

    # Hardware arguments
    parser.add_argument("--strategy", type=str, default="deepspeed", help="Training strategy")
    parser.add_argument("--devices", type=int, default=4, help="Number of GPU devices")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir", type=str, default=".", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="Save checkpoint every N epochs (0 to disable)"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()
    main(args)

import argparse
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
import webdataset as wds
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
)

from codontransformer2.dataset import MaskedTokenizerCollator


class TrainHarness(pl.LightningModule):
    """
    PyTorch Lightning module for training CodonTransformer with species-specific embeddings.

    This harness wraps a ModernBERT model for masked language modeling on codon sequences,
    adding species-specific embeddings to enable organism-aware codon optimization.

    Args:
        model: ModernBertForMaskedLM model instance
        n_species: Total number of species/organisms to support
        learning_rate: Maximum learning rate for training
        warmup_fraction: Fraction of total steps for linear warmup (default: 0.1)
        learning_rate_decay: Factor to decay the learning rate by (default: 0.1)

    Attributes:
        model: The underlying transformer model
        species_embed: Embedding layer for species-specific representations
        learning_rate: Learning rate for optimization
        warmup_fraction: Fraction of training for warmup phase
        learning_rate_decay: Factor to decay the learning rate by
    """

    def __init__(self, model, n_species, learning_rate, warmup_fraction, learning_rate_decay):
        super().__init__()
        self.model = model
        self.species_embed = nn.Embedding(n_species, model.config.hidden_size)
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.learning_rate_decay = learning_rate_decay

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step with species-conditioned masked language modeling.

        Args:
            batch: Dictionary containing:
                - input_ids: Tokenized codon sequences [B, L]
                - species_id: Species identifiers [B]
                - attention_mask: Attention mask [B, L]
                - labels: Target labels for masked tokens [B, L]
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Loss value for this batch
        """
        # Get token embeddings and add species-specific information
        token_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        sp_vec = self.species_embed(batch["species_id"])  # [B, H]
        sp_vec = sp_vec.unsqueeze(1).expand_as(token_embeds)  # [B, L, H]
        token_embeds = token_embeds + sp_vec

        # Forward pass through the model
        outputs = self.model(
            inputs_embeds=token_embeds,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Log metrics
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return outputs.loss

    def configure_optimizers(self):
        """
        Configure AdamW optimizer with linear warmup and cosine decay schedule.

        Uses 10% linear warmup followed by 90% cosine annealing decay to 0.1 of max LR.

        Returns:
            Tuple[List, List]: A tuple containing:
                - List of optimizers (AdamW)
                - List of scheduler configs with step-level updates
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup_fraction * total_steps)
        decay_steps = total_steps - warmup_steps

        print(f"Optimizer configured with LR: {self.learning_rate}")
        print(
            f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Decay steps: {decay_steps}"
        )

        # Linear warmup from near-zero to max LR
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine decay from max LR to 0.1 * max LR
        decay = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.learning_rate * self.learning_rate_decay,
        )

        # Combine warmup and decay
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class EpochCheckpoint(pl.Callback):
    """
    PyTorch Lightning callback for saving model checkpoints at epoch intervals.

    Saves checkpoints at regular epoch intervals, including epoch 0 (initialization).

    Args:
        save_interval: Save checkpoint every N epochs (e.g., 1 for every epoch)
        checkpoint_dir: Directory path where checkpoints will be saved

    Attributes:
        checkpoint_dir: Directory for saving checkpoints
        save_interval: Interval between checkpoint saves
    """

    def __init__(self, save_interval, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch to save checkpoints.

        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The LightningModule being trained

        Returns:
            None
        """
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_interval == 0 or current_epoch == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}\n")


def main(args):
    """
    Main training function for CodonTransformer2.
    """
    # Set random seed and configure PyTorch for optimal performance
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Initialize codon tokenizer with special tokens
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

    # Configure ModernBERT model for masked language modeling
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        max_position_embeddings=args.max_length,
        attn_implementation="flash_attention_2",
    )
    model = ModernBertForMaskedLM(config=config)

    # Wrap model with species-aware training harness
    harnessed_model = TrainHarness(
        model,
        n_species=args.num_organisms + args.extra_organisms,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        learning_rate_decay=args.learning_rate_decay,
    )

    # Set up WebDataset pipeline for distributed training
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

    # Configure callbacks
    callbacks = []
    if args.save_interval > 0:
        callbacks.append(EpochCheckpoint(args.save_interval, args.checkpoint_dir))

    # Initialize PyTorch Lightning trainer with distributed training support
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

    # Start training
    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training harness.")

    # Data arguments
    parser.add_argument("--dataroot", type=str, required=True, help="Root directory for data")
    parser.add_argument(
        "--shard_pattern",
        type=str,
        default="shard-{000000..003863}.tar",
        help="Pattern for data shards",
    )
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default="data/codon_transformer_tokenizer.json",
        help="Path to tokenizer file",
    )

    # Model arguments
    parser.add_argument("--num_organisms", type=int, default=4742, help="Number of organisms")
    parser.add_argument("--extra_organisms", type=int, default=2000, help="Extra organism slots")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--learning_rate_decay", type=float, default=0.1, help="Learning rate decay"
    )
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="Warmup fraction")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument(
        "--limit_train_batches", type=int, default=400_000, help="Limit training batches"
    )
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")

    # Hardware arguments
    parser.add_argument("--strategy", type=str, default="deepspeed", help="Training strategy")
    parser.add_argument("--devices", type=int, default=4, help="Number of GPU devices")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision")

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir", type=str, default=".", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1, help="Save checkpoint every N epochs (0 to disable)"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()
    main(args)

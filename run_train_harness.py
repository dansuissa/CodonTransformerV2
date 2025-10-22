import argparse
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    ModernBertConfig,
    ModernBertForMaskedLM,
)

from codon_transformer_2.collators import MaskedTokenizerCollator

NUM_ORGANISMS = 4742
EXTRA_ORGANISMS = 2000
LEARNING_RATE = 1e-4
WARMUP_FRACTION = 0.1
MAX_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = 8
DEBUG = True


class TrainHarness(pl.LightningModule):
    def __init__(self, model, n_species):
        super().__init__()
        self.model = model
        self.species_embed = nn.Embedding(n_species, model.config.hidden_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=LEARNING_RATE,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=WARMUP_FRACTION,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        token_embeds = self.model.get_input_embeddings()(batch["input_ids"])
        sp_vec = self.species_embed(batch["species_id"])          # [B, H]
        sp_vec = sp_vec.unsqueeze(1).expand_as(token_embeds)      # [B, L, H]
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
    def __init__(self, save_interval):
        super().__init__()
        self.checkpoint_dir = "."
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_interval == 0 or current_epoch == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}\n")


def main(args):
    pl.seed_everything(123)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="data/codon_transformer_tokenizer.json",
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        max_len=1024,
    )

    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        max_position_embeddings=1024,
        attn_implementation="flash_attention_2",
    )
    model = ModernBertForMaskedLM(config=config)
    harnessed_model = TrainHarness(model, n_species=NUM_ORGANISMS + EXTRA_ORGANISMS)

    dataroot = os.environ.get("DATAROOT", "/lustre/fsn1/projects/rech/nef/unh87ms/webdataset")
    shard_pattern = os.path.join(dataroot, "shard-{000000..003863}.tar")
    train_data = wds.WebDataset(
        shard_pattern,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
    )
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )

    trainer = pl.Trainer(
        default_root_dir=".",
        strategy="deepspeed",
        accelerator="gpu",
        devices=4,
        precision="bf16-mixed",
        max_epochs=MAX_EPOCHS,
        deterministic=False,
        enable_checkpointing=True,
        limit_train_batches=400_000,
        log_every_n_steps=10,
    )

    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training harness.")
    args = parser.parse_args()
    main(args)

#!/bin/bash

# Training script for CodonTransformer2
# Edit the variables below to configure your training run

# Data configuration
DATAROOT="/lustre/fsn1/projects/rech/nef/unh87ms/webdataset"
SHARD_PATTERN="shard-{000000..003863}.tar"
TOKENIZER_FILE="data/codon_transformer_tokenizer.json"

# Model configuration
NUM_ORGANISMS=4742
EXTRA_ORGANISMS=2000
MAX_LENGTH=1024

# Training configuration
LEARNING_RATE=1e-4
WARMUP_FRACTION=0.1
MAX_EPOCHS=5
BATCH_SIZE=32
NUM_WORKERS=8
LIMIT_TRAIN_BATCHES=400000
LOG_EVERY_N_STEPS=10

# Hardware configuration
STRATEGY="deepspeed"
DEVICES=4
PRECISION="bf16-mixed"

# Checkpoint configuration
CHECKPOINT_DIR="."
SAVE_INTERVAL=1

# Other configuration
SEED=123

# Run training
python train.py \
    --dataroot "$DATAROOT" \
    --shard-pattern "$SHARD_PATTERN" \
    --tokenizer-file "$TOKENIZER_FILE" \
    --num-organisms "$NUM_ORGANISMS" \
    --extra-organisms "$EXTRA_ORGANISMS" \
    --max-length "$MAX_LENGTH" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-fraction "$WARMUP_FRACTION" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --limit-train-batches "$LIMIT_TRAIN_BATCHES" \
    --log-every-n-steps "$LOG_EVERY_N_STEPS" \
    --strategy "$STRATEGY" \
    --devices "$DEVICES" \
    --precision "$PRECISION" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-interval "$SAVE_INTERVAL" \
    --seed "$SEED"

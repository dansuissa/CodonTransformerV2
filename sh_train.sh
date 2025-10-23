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
MAX_LENGTH=2048

# Training configuration
LEARNING_RATE=1e-4
LEARNING_RATE_DECAY=0.1
WARMUP_FRACTION=0.1
WEIGHT_DECAY=0.01
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
SEED=23

# Run training
python train.py \
    --dataroot "$DATAROOT" \
    --shard_pattern "$SHARD_PATTERN" \
    --tokenizer_file "$TOKENIZER_FILE" \
    --num_organisms "$NUM_ORGANISMS" \
    --extra_organisms "$EXTRA_ORGANISMS" \
    --max_length "$MAX_LENGTH" \
    --learning_rate "$LEARNING_RATE" \
    --learning_rate_decay "$LEARNING_RATE_DECAY" \
    --warmup_fraction "$WARMUP_FRACTION" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --limit_train_batches "$LIMIT_TRAIN_BATCHES" \
    --log_every_n_steps "$LOG_EVERY_N_STEPS" \
    --strategy "$STRATEGY" \
    --devices "$DEVICES" \
    --precision "$PRECISION" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --save_interval "$SAVE_INTERVAL" \
    --seed "$SEED"

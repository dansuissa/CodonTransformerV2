#!/bin/bash
#SBATCH -A nef@h100
#SBATCH --job-name=ct2_s5_8
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=preflight_logs/40_stage5_fullmask_mix_8h100_%j.out
#SBATCH --error=preflight_logs/40_stage5_fullmask_mix_8h100_%j.err

set -euo pipefail

cd /lustre/fswork/projects/rech/nef/unh87ms/CodonTransformerV2
export SCRATCH=/lustre/fsn1/projects/rech/nef/unh87ms

module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=$SCRATCH/.cache
export HF_HOME=$SCRATCH/huggingface
export TORCH_HOME=$SCRATCH/torch
export TMPDIR=/tmp/ct2_stage5_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"

CKPT=/lustre/fsn1/projects/rech/nef/unh87ms/ct2_final/h100_8gpu_stage4_len2048_from_stage3e14_bs56_w2_acc3_lr5e6/epoch_7.ckpt
OUTDIR=/lustre/fsn1/projects/rech/nef/unh87ms/ct2_final/h100_8gpu_stage5_fullmask30_from_stage4e7_bs56_w2_acc3_lr2e6_${SLURM_JOB_ID}

echo "STAGE5_8H100_START $(date)"
echo "CKPT=$CKPT"
echo "OUTDIR=$OUTDIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-NA}"

srun python train_updated.py \
  --dataroot /lustre/fsn1/projects/rech/nef/unh87ms/codon_project/data/CT2_data/train_set_shard \
  --shard_pattern "shard-{000000..005873}.tar.gz" \
  --tokenizer_file /lustre/fswork/projects/rech/nef/unh87ms/CodonTransformerV2/CodonTransformerTokenizer.json \
  --species_to_id_path /lustre/fsn1/projects/rech/nef/unh87ms/codon_project/data/CT2_data/species_to_id.json \
  --test_json_dir /lustre/fsn1/projects/rech/nef/unh87ms/codon_project/data/CT2_data/test_set \
  --test_batch_size 2 \
  --limit_test_batches 2000 \
  --check_test_every_n_epoch 1 \
  --do_species_ablations_every 10 \
  --num_organisms 26678 \
  --extra_organisms 2000 \
  --unknown_species_id 26678 \
  --species_dropout_prob 0.0 \
  --mlm_probability 0.15 \
  --full_mask_probability 0.30 \
  --max_length 2048 \
  --attn_implementation sdpa \
  --learning_rate 2e-6 \
  --warmup_fraction 0.05 \
  --learning_rate_decay 0.2 \
  --weight_decay 0.01 \
  --max_epochs 4 \
  --batch_size 56 \
  --num_workers 8 \
  --accumulate_grad_batches 3 \
  --limit_train_batches 8000 \
  --log_every_n_steps 50 \
  --eval_topk 5 \
  --synonym_topk 1 \
  --shardshuffle 1000 \
  --accelerator gpu \
  --devices 4 \
  --num_nodes 2 \
  --strategy ddp \
  --precision bf16-mixed \
  --checkpoint_dir "$OUTDIR" \
  --save_interval 1 \
  --seed 123 \
  --init_from_checkpoint "$CKPT"

echo "STAGE5_8H100_DONE $(date)"
ls -lh "$OUTDIR" || true

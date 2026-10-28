#!/bin/bash
# Train one model on Rangpur. Submit from this folder:
#   sbatch train.slurm unet3d                 # full run
#   sbatch train.slurm unet2d
#   sbatch --time=00:15:00 train.slurm unet3d --epochs 2   # quick check on real data
# Extra arguments after the model name are passed to train.py.
#SBATCH --job-name=hipmri
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
# No --mem: comp3710 nodes are registered in Slurm with 1 MB of memory, so any --mem request
# leaves the job pending forever. Jobs that start on this partition all show mem=1M.
# --account=comp3710: the default personal account is rejected by the partition (PartitionConfig).
#SBATCH --output=%x_%j.out

MODEL=${1:?usage: sbatch train.slurm <unet3d|unet2d> [extra train.py args]}
shift
EXTRA_ARGS=("$@")
set --  # clear positional args: a sourced script (conda's activate) would otherwise receive them

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
git log --oneline -1  # record which code version produced this run

source "$HOME/miniconda3/bin/activate"
conda activate torch
echo "python: $(which python)"

# --max-hours leaves time inside the 3 h limit for the test pass after training
python train.py --model "$MODEL" --amp --max-hours 2.5 --out "runs/${MODEL}_${SLURM_JOB_ID}" "${EXTRA_ARGS[@]}"

echo "Finished $(date)"

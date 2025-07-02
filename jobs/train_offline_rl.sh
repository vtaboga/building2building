#!/bin/bash
#SBATCH --job-name=offline_rl_training
#SBATCH --output=logs/offline_rl_%j.out  
#SBATCH --error=logs/offline_rl_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Activate environment (adjust path as needed)
source ~/.bashrc
conda activate building2building  # or your environment name

# Change to project directory
cd /path/to/Building2Building  # Update this path

# Create logs directory if it doesn't exist
mkdir -p logs

# Example training commands - customize as needed

# Train CQL on a dataset
echo "Training CQL with offline dataset..."
python scripts/train_offline_rl.py \
    --state AL \
    --county Mobile \
    --building-id building_001 \
    --weather Mobile_AL.epw \
    --dataset-path data/offline_datasets/baseline_dataset.npz \
    --algorithm cql \
    --epoch 1000 \
    --batch-size 256 \
    --lr 3e-4 \
    --cql-weight 1.0 \
    --temperature 1.0 \
    --track \
    --save-model \
    --seed 42

echo "Offline RL training completed!" 
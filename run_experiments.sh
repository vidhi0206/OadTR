#!/bin/bash
#SBATCH --job-name=vit_experiments
#SBATCH --output=vit_experiments.%j.out
#SBATCH --error=vit_experiments.%j.err
#SBATCH --partition=longrun  
#SBATCH --cpus-per-task=1        # CPU cores per GPU for faster data loading
#SBATCH --gres=gpu:1             # number of GPUs to use
#SBATCH --ntasks-per-node=1      # one task per GPU
#SBATCH --time=13-23:00:00           # max runtime
#SBATCH --mem=30G                # memory per node

# Load conda
module load tuni/miniforge3/24.9.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate OadTR

# Move to project folder
cd ~/tdropreg/T2T-ViT

# Prevent picking up old user-site packages
export PYTHONNOUSERSITE=1

python main.py --num_layers 3 --decoder_layers 5 --enc_layers 64  --feature anet --dim_feature 3072

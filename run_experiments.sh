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
conda activate Oadtr

# Move to project folder
cd ~/OadTR/OadTR

# Prevent picking up old user-site packages
export PYTHONNOUSERSITE=1
NUM_GPU=${SLURM_NTASKS_PER_NODE:-1} 
 # Adjust per-GPU batch to avoid OOM 
GLOBAL_BATCH_SIZE=128
PER_GPU_BATCH=$(($GLOBAL_BATCH_SIZE / $NUM_GPU)) 
echo "Running on $NUM_GPU GPUs with per-GPU batch size $PER_GPU_BATCH"
GPU_ID=0
DEVICE="cuda:${GPU_ID}"
LAMBDAS=(0.01)
DATASETS=("t_imgnet")
DR_MHA_LIST=("V" "none" "K" "Q" )
DR_MLP_MODES=(2 1)
DROP_MHA_LIST=("drop_none")
DROP_RATIOS=(0.2)

# Adjust per-GPU batch to avoid OOM


for DROP_MHA in "${DROP_MHA_LIST[@]}"; do
    for DR_MHA in "${DR_MHA_LIST[@]}"; do
        for DROP_RATIO in "${DROP_RATIOS[@]}"; do
            for LAMBDA in "${LAMBDAS[@]}"; do
                for MLP_MODE in "${DR_MLP_MODES[@]}"; do

                    python main.py \
                    --num_layers 2 \
                    --enc_layers 64 \
                    --feature Anet2016_feature_v2 \
                    --drop_mha ${DROP_MHA} \
                    --dr_mha ${DR_MHA} \
                    --dr_mlp_mode ${MLP_MODE} \
                    --Lambda ${LAMBDA} \
                    --drop ${DROP_RATIO} \
                    --attn-drop-rate ${DROP_RATIO} \


                done
            done
        done
    done
done
#!/bin/bash -e
#SBATCH --job-name=fsw_1
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/chuyen-rang/fsw_1.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/chuyen-rang/fsw_1.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io
prep_env fairsw
cd /lustre/scratch/client/vinai/users/hainn14/chuyen-rang

weight_fsw_values=(0.5)
methods=(EFBSW FBSW lowerboundFBSW BSW)
obsw_weights=(0.1 1.0 10.0)
num_epochs=300
seed_id=42
batch_size=1000
batch_size_test=128
distribution="circle"
optimizer="rmsprop"
lr=0.001
saved_model_interval=50
alpha=0.9
datadir="data"
outdir="result"
weight_swd=8.0

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        python3 train.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir "$datadir" \
            --outdir "$outdir" \
            --distribution "$distribution" \
            --epochs "$num_epochs" \
            --optimizer "$optimizer" \
            --lr "$lr" \
            --alpha "$alpha" \
            --batch-size "$batch_size" \
            --batch-size-test "$batch_size_test" \
            --seed "$seed_id" \
            --weight_swd "$weight_swd" \
            --weight_fsw "$weight_fsw" \
            --method "$method" \
            --saved-model-interval "$saved_model_interval"
    done

    for lmbd in "${obsw_weights[@]}"; do
        python3 train.py \
            --dataset mnist \
            --num-classes 10 \
            --datadir "$datadir" \
            --outdir "$outdir" \
            --distribution "$distribution" \
            --epochs "$num_epochs" \
            --optimizer "$optimizer" \
            --lr "$lr" \
            --alpha "$alpha" \
            --batch-size "$batch_size" \
            --batch-size-test "$batch_size_test" \
            --seed "$seed_id" \
            --weight_swd "$weight_swd" \
            --weight_fsw "$weight_fsw" \
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --saved-model-interval "$saved_model_interval"
    done
done
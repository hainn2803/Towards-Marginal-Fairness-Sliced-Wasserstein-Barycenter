#!/bin/bash

# Define parameters
seed_id=28
weight_fsw_values=(0.5)
methods=(EFBSW FBSW lowerboundFBSW BSW)
obsw_weights=(0.1 1.0 10.0)
checkpoint=250

# Initial plot
python3 plot_latent.py \
    --no-cuda \
    --dataset mnist \
    --num-classes 10 \
    --datadir data \
    --outdir result \
    --distribution circle \
    --batch-size 100 \
    --batch-size-test 100 \
    --lr 0.001 \
    --seed "$seed_id" \
    --weight_fsw 0.0 \
    --checkpoint-period "$checkpoint" \
    --method None

# for weight_fsw in "${weight_fsw_values[@]}"; do
#     for method in "${methods[@]}"; do
#         python3 plot_latent.py \
#             --dataset mnist \
#             --num-classes 10 \
#             --datadir data \
#             --outdir result \
#             --distribution circle \
#             --batch-size 100 \
#             --batch-size-test 100 \
#             --lr 0.001 \
#             --seed "$seed_id" \
#             --weight_fsw "$weight_fsw" \
#             --checkpoint-period "$checkpoint" \
#             --method "$method"
#     done
# done

# for weight_fsw in "${weight_fsw_values[@]}"; do
#     for lmbd in "${obsw_weights[@]}"; do
#         python3 plot_latent.py \
#             --dataset mnist \
#             --num-classes 10 \
#             --datadir data \
#             --outdir result3 \
#             --distribution circle \
#             --batch-size 100 \
#             --batch-size-test 100 \
#             --lr 0.001 \
#             --seed "$seed_id" \
#             --weight_fsw "$weight_fsw" \
#             --checkpoint-period "$checkpoint" \
#             --method OBSW \
#             --lambda-obsw "$lmbd"
#     done
# done

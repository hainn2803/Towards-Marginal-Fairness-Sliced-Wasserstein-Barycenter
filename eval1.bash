#!/bin/bash

methods=(EFBSW FBSW)
# obsw_weights=(0.1 1.0 10.0)
checkpoint_periods=(300 250 200 150 100 50)
fsw=0.5
datadir=data
outputdir=result
distribution=circle
batch_size=128
batch_size_test=128
seed=28
lr=0.001
gpu_id=0

for ckp in "${checkpoint_periods[@]}"; do

    for ((i=1; i<=10; i++)); do

        # CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator.py \
        #     --no-cuda \
        #     --dataset mnist \
        #     --num-classes 10 \
        #     --datadir "$datadir" \
        #     --outdir "$outputdir" \
        #     --distribution "$distribution" \
        #     --batch-size "$batch_size" \
        #     --batch-size-test "$batch_size_test" \
        #     --lr "$lr" \
        #     --seed "$seed" \
        #     --weight_fsw 0.0 \
        #     --checkpoint-period "$ckp" \
        #     --method None

        for method in "${methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator.py \
                --no-cuda \
                --dataset mnist \
                --num-classes 10 \
                --datadir "$datadir" \
                --outdir "$outputdir" \
                --distribution "$distribution" \
                --batch-size "$batch_size" \
                --batch-size-test "$batch_size_test" \
                --lr "$lr" \
                --seed "$seed" \
                --weight_fsw "$fsw" \
                --checkpoint-period "$ckp" \
                --method "$method"
        done

        # for lmbd in "${obsw_weights[@]}"; do
        #     CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator.py \
        #         --no-cuda \
        #         --dataset mnist \
        #         --num-classes 10 \
        #         --datadir "$datadir" \
        #         --outdir "$outputdir" \
        #         --distribution "$distribution" \
        #         --batch-size "$batch_size" \
        #         --batch-size-test "$batch_size_test" \
        #         --lr "$lr" \
        #         --seed "$seed" \
        #         --weight_fsw "$fsw" \
        #         --method OBSW \
        #         --lambda-obsw "$lmbd" \
        #         --checkpoint-period "$ckp"
        # done

    done
done


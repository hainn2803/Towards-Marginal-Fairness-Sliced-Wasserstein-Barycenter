obsw_weights=(0.1 1.0 10.0)
checkpoint_periods=(150 100 50)
fsw=0.5
datadir=data
outputdir=result
distribution=circle
batch_size=128
batch_size_test=128
seed=42
lr_values=(0.0005 0.0001)
dataset=mnist

gpu_id=0

for lr in "${lr_values[@]}"; do
    for ckp in "${checkpoint_periods[@]}"; do
        for ((i=1; i<=20; i++)); do
            for lmbd in "${obsw_weights[@]}"; do
                CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator.py \
                    --dataset "$dataset" \
                    --num-classes 10 \
                    --datadir "$datadir" \
                    --outdir "$outputdir" \
                    --distribution "$distribution" \
                    --batch-size "$batch_size" \
                    --batch-size-test "$batch_size_test" \
                    --lr "$lr" \
                    --seed "$seed" \
                    --weight_fsw "$fsw" \
                    --method OBSW \
                    --lambda-obsw "$lmbd" \
                    --checkpoint-period "$ckp"
            done
        done
    done
done


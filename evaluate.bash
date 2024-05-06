weight_fsw_values=(0.5)
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 10.0)
checkpoint_periods=(300 300 300 300 300 300 300 300 300 300)
fsw=0.5
datadir=data
outputdir=result
distribution=circle
batch_size=128
batch_size_test=128
seed=42
lr=0.001

for ckp in "${checkpoint_periods[@]}"; do
    python3 evaluator.py \
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
        --weight_fsw 0.0 \
        --checkpoint-period "$ckp" \
        --method None

    for method in "${methods[@]}"; do
        python3 evaluator.py \
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

    for lmbd in "${obsw_weights[@]}"; do
        python3 evaluator.py \
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
            --method OBSW \
            --lambda-obsw "$lmbd" \
            --checkpoint-period "$ckp"
    done
done

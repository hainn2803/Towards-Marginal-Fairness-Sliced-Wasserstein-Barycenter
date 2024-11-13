lr_values=(0.001)
methods=(lowerboundFBSW BSW)
checkpoint_periods=(150 100 50)
fsw=0.5
datadir=data
outputdir=result
distribution=circle
batch_size=128
batch_size_test=128
seed=42

gpu_id=2

for lr in "${lr_values[@]}"; do
    for ckp in "${checkpoint_periods[@]}"; do
        for ((i=1; i<=5; i++)); do

            CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator.py \
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
                --method None \
                --checkpoint-period "$ckp"

            for method in "${methods[@]}"; do
                CUDA_VISIBLE_DEVICES="$gpu_id" python3 evaluator.py \
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
        done
    done
done

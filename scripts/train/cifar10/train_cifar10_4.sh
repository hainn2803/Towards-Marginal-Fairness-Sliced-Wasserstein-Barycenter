weight_fsw_values=(0.5)
methods=(lowerboundFBSW)
num_epochs=500
seed_id=42
batch_size=1000
batch_size_test=128
distribution="uniform"
optimizer="adam"
embed_size=48
lr=0.0005
saved_model_interval=100
beta1=0.9
beta2=0.999
datadir="data"
outdir="result"
weight_swd=8.0

gpu_id=3


for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
            --dataset cifar10 \
            --num-classes 10 \
            --datadir "$datadir" \
            --outdir "$outdir" \
            --distribution "$distribution" \
            --epochs "$num_epochs" \
            --optimizer "$optimizer" \
            --lr "$lr" \
            --beta1 "$beta1" \
            --beta2 "$beta2" \
            --batch-size "$batch_size" \
            --batch-size-test "$batch_size_test" \
            --embedding-size "$embed_size" \
            --seed "$seed_id" \
            --weight_swd "$weight_swd" \
            --weight_fsw "$weight_fsw" \
            --method "$method" \
            --saved-model-interval "$saved_model_interval"
    done
done
num_epochs=500
seed_id=42
batch_size=1000
batch_size_test=128
distribution="uniform"
optimizer="adam"
lr=0.0005
saved_model_interval=100
beta1=0.9
beta2=0.999
datadir="data"
outdir="result"
weight_swd=8.0

gpu_id=5

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
    --seed "$seed_id" \
    --weight_swd "$weight_swd" \
    --weight_fsw 0.0 \
    --method None \
    --saved-model-interval "$saved_model_interval"

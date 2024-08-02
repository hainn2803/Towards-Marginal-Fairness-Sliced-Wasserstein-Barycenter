num_epochs=800
seed_id=1
batch_size=1000
batch_size_test=128
distribution="uniform"
optimizer="adam"
lr=0.001
saved_model_interval=100
beta1=0.9
beta2=0.999
datadir="data"
outdir="result"
weight_swd=8.0
embed=32

gpu_id=3

CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
    --dataset stl10 \
    --num-classes 10 \
    --datadir "$datadir" \
    --outdir "$outdir" \
    --distribution "$distribution" \
    --embedding-size "$embed" \
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

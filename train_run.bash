weight_fsw_values=(0.5)
methods=(EFBSW FBSW lowerboundFBSW OBSW BSW)
obsw_weights=(0.1 1.0 10.0)
num_epochs=300
seed_id=42
gpu_id=6
batch_size=1000
batch_size_test=128
distribution="circle"
optimizer="rmsprop"
lr=0.001
saved_model_interval=1
alpha=0.9
datadir="data"
outdir="result"
weight_swd=8.0

CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
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
  --weight_fsw 0.0 \
  --method None \
  --saved-model-interval "$saved_model_interval" \

for weight_fsw in "${weight_fsw_values[@]}"; do
    for method in "${methods[@]}"; do
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
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
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 train.py \
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

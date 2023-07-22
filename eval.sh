dir=train_2023-07-21_23:26:44

for weights in exp/"$dir"/checkpoints/*.pth; do
  echo "$weights"
  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset_name ethanol --cutoff 4 --batch_size 256 \
    --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
    --use_invariance \
    --checkpoint "$weights" || exit
  echo ""
done

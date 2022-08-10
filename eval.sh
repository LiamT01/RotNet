for weights in /GPFS/data/hongweitu/rotation/exp/train_2022-08-10_01:45:00/*.pth; do
  echo "$weights"
  CUDA_VISIBLE_DEVICES=8 python evaluate.py \
    --dataset_name qm7 --cutoff 6 --batch_size 512 \
    --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
    --use_invariance \
    --checkpoint "$weights" || exit
done
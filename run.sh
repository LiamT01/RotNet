CUDA_VISIBLE_DEVICES=4 python train.py \
  --dataset_name qm7 --cutoff 6 --lr 0.001 --batch_size 32 --epochs 10000 \
  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
  --use_invariance
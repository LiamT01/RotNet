#CUDA_VISIBLE_DEVICES=0,1,2,4,6,7,8,9 python train.py \
#  --distributed --num_nodes 1 --num_gpus_per_node 8 --node_rank 0 \
#  --dataset_name qm7 --cutoff 6 --lr 0.001 --batch_size 32 --epochs 10000 \
#  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
#  --use_invariance

CUDA_VISIBLE_DEVICES=5 python train.py \
  --dataset_name qm7 --cutoff 6 --lr 0.001 --batch_size 256 --epochs 10000 \
  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
  --use_invariance

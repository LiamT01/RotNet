# Distributed training

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python train.py \
#  --distributed --num_nodes 1 --num_gpus_per_node 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 8888 \
#  --dataset_name uracil --cutoff 4 --lr 0.001 --batch_size 768 --epochs 10000 \
#  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
#  --use_invariance

# Single GPU

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name ethanol --cutoff 4 --lr 0.01 --batch_size 32 --epochs 100000 \
  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
  --use_invariance

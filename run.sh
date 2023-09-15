# Distributed training

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
#  --distributed --num_nodes 1 --num_gpus_per_node 4 --node_rank 0 --master_addr 0.0.0.0 --master_port 8888 \
#  --dataset_name electronic_materials --cutoff 0.6 --lr 0.001 --batch_size 64 --epochs 1600 \
#  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
#  --use_invariance

# Single GPU

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_name electronic_materials --cutoff 0.6 --lr 0.001 --batch_size 256 --epochs 1600 \
  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
  --use_invariance

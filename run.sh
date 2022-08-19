CUDA_VISIBLE_DEVICES=1,2,4,5,6,7,8,9 python train.py \
  --distributed --num_nodes 1 --num_gpus_per_node 8 --node_rank 0 --master_addr 0.0.0.0 --master_port 8888\
  --dataset_name amino_acid --cutoff 4 --lr 0.001 --batch_size 64 --epochs 10000 \
  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
#  --use_invariance

#CUDA_VISIBLE_DEVICES=5 python train.py \
#  --dataset_name band_gap --cutoff 0.6 --lr 0.001 --batch_size 256 --epochs 10000 \
#  --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
#  --use_invariance

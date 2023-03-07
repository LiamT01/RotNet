for weights in exp/train_2023-03-07_15:05:20/checkpoints/*.pth; do
  echo "$weights"
  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset_name protein_fragments --cutoff 4 --batch_size 512 \
    --num_layers 5 --gaussian_num_steps 50 --x_size 92 --hidden_size 512 \
    --use_invariance \
    --checkpoint "$weights" || exit
  echo ""
done

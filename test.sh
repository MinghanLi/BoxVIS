CUDA_VISIBLE_DEVICES=1 python train_net_boxvis.py \
  --num-gpus 1 \
  --config-file configs/boxvis_R50_bs16_test_ytvis21.yaml \
  --eval-only \

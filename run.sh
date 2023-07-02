CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net_boxvis.py --num-gpus 8 \
  --dist-url tcp://127.0.0.1:50153 \
  --config-file configs/boxvis_R50_bs16.yaml \
  TEST.EVAL_PERIOD 0 \



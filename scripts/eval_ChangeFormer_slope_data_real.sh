#!/usr/bin/env bash

gpus=0,1

data_name=slope_data_real
net_G=ChangeFormerV6 #This is the best version
split=test
vis_root=vis
project_name=ChangeFormer_slope
checkpoints_root=./pretrained_changeformer
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=0,1 python eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}



#!/bin/bash
#########################################################################
# Author: Bin Liang
# mail: bin.liang@stu.hit.edu.cn
#########################################################################

CUDA_VISIBLE_DEVICES=2 python train_bert.py --model_name intergcn_bert --dataset rest15 --num_epoch 20 --lr 2e-5 --batch_size 16 --seed 776

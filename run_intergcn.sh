#!/bin/bash
#########################################################################
# Author: Bin Liang
# mail: bin.liang@stu.hit.edu.cn
#########################################################################

CUDA_VISIBLE_DEVICES=1 python train.py --model_name intergcn --dataset rest14 --save True --learning_rate 1e-3 --seed 29 --batch_size 16

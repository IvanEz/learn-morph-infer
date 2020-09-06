#!/bin/bash

python3 main.py --is_3d=True --dataset 'tumor_mparam' --res_x=64 --res_y=64 --res_z=64 --num-conv 4 --batch_size=16 --num_worker=8 --log_ep=5 --test_ep=5 --use_curl False  --arch 'alternative' --phys_loss False --is_train True --gpu_id="2" --max_epoch 30 --tag 'clipped_0.001_geom_3_extracsf'


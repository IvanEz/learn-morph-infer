#!/bin/bash

python3 main.py --is_3d=True --dataset 'samples_extended' --res_x=128 --res_y=128 --res_z=128 --filters=32 --num-conv 4 --batch_size=16 --num_worker=8 --log_ep=5 --test_ep=5 --use_curl False  --arch 'alternative' --phys_loss False --is_train True --gpu_id="0" --max_epoch 30 --lr_max 0.01 --lr_min 0.00025 --beta1 0.9 --tag 'inverse_test_lr0.01-0.00025_stdbt_4x4x4_numconv4_l2' --valid_dataset_dir '/mnt/Drive2/ivan/data/valid' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/testingstuff' --random_seed 789


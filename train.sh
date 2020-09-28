#!/bin/bash

python3 main.py --is_3d=True --dataset 'Dataset' --res_x=128 --res_y=128 --res_z=128 --filters=32 --num-conv 4 --batch_size=16 --num_worker=8 --log_ep=2 --test_ep=2 --use_curl False  --arch 'alternative' --phys_loss False --is_train True --gpu_id="1" --max_epoch 10 --lr_max 0.001 --lr_min 0.0001 --beta1 0.9 --tag 'inverse_test_lr0.001-0.0001_stdbt_4x4x4_numconv4_l2_20k_10ep_normrange100_fc92_lrelu' --valid_dataset_dir '/mnt/Drive2/ivan/data/valid' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/samples_extended_copy' --random_seed 16398


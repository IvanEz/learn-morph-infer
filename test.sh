#!/bin/bash

python3 main.py --is_train=False --load_path=/mnt/Drive2/ivan_kevin/log/samples_extended/0926_105118_alternative_inverse_test_lr0.009-0.0007_stdbt_4x4x4_numconv3_l2 --test_batch_size=1 --is_3d=True --dataset='tumor_mparam' --res_x=128 --res_y=128 --res_z=128 --batch_size=4 --num_worker=1 --num-conv 4 --use_curl False --arch 'alternative' --phys_loss False --gpu_id="4" --data_dir=/mnt/Drive2/ivan_kevin/testsamples --inf_save=/mnt/Drive2/ivan_kevin/testsamples_results

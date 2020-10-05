#!/bin/bash

#python3 main.py --random_seed=12345 --endtraindata=12 --startvaldata=12 --endvaldata=16 --outputparams=6 --filters=32 --num_conv=3 --log_step=1 --batch_size=4 --num_worker=8 --is_train True --gpu_id="1" --max_epoch=80 --lr_max=0.0003 --lr_min=0.00005 --beta1=0.9 --tag 'inverse_notthresholded_sanity_benchmark' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/samples_extended_copy' --dataset 'Dataset'
python3 main.py --random_seed=32126 --fchdepth=0 --fcsize=3 --starttraindata=0 --endtraindata=40000 --startvaldata=80000 --endvaldata=84000 --outputparams=3 --filters=2 --num_conv=7 --log_step=32 --batch_size=32 --num_worker=8 --is_train True --gpu_id="1" --max_epoch=4 --lr_max=0.0001 --lr_min=0.00000025 --beta1=0.9 --tag 'inverse_test_notthresholded_Dw_rho_T' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/samples_extended_copy' --dataset 'Dataset'


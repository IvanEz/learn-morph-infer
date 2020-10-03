#!/bin/bash

python3 main.py --random_seed=12345 --endtraindata=12 --startvaldata=12 --endvaldata=16 --outputparams=6 --filters=32 --num_conv=3 --log_step=1 --batch_size=4 --num_worker=8 --is_train True --gpu_id="1" --max_epoch=80 --lr_max=0.0003 --lr_min=0.00005 --beta1=0.9 --tag 'inverse_notthresholded_sanity_benchmark' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/samples_extended_copy' --dataset 'Dataset'


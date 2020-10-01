#!/bin/bash

python3 main.py --random_seed=84905 --outputparams=6 --filters=32 --num_conv=3 --log_step=30 --batch_size=12 --num_worker=8 --is_train True --gpu_id="1" --max_epoch=20 --lr_max=0.0003 --beta1=0.9 --tag 'inverse_test_notthresholded_sanity' --log_dir '/mnt/Drive2/ivan_kevin/log' --data_dir '/mnt/Drive2/ivan_kevin/samples_extended_copy' --dataset 'Dataset'


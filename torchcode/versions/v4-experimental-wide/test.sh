#!/bin/bash

python3 main.py --batch_size 64 --loaddir /mnt/Drive2/ivan_kevin/log/torchimpl/2211-18-50-33-v4-exper-wide-pet-maxpool-nobn/bestval-model.pt --num_workers 4 --startval 80000 --endval 88000 --gpu 5

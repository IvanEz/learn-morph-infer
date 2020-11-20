#!/bin/bash

#python3 main.py --isnewsave --purpose test --batch_size 1 --num_workers 2 --starttrain 0 --endtrain 4 --startval 0 --endval 4 --dropoutrate 0.0 --lr 0.001 --gpu 6
python3 main.py --isnewsave --gpu 5 --purpose pet-maxpool --batch_size 128 --num_workers 4 --starttrain 0 --endtrain 80000 --startval 80000 --endval 88000 --dropoutrate 0.0 --lr 0.0006 --lr_scheduler_rate 0.99991 --weight_decay_sgd 0.01

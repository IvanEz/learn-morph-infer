#!/bin/bash

#python3 main.py --isnewsave --purpose test --batch_size 1 --num_workers 2 --starttrain 0 --endtrain 4 --startval 0 --endval 4 --dropoutrate 0.0 --lr 0.001 --gpu 6
python3 main.py --isnewsave --gpu 5 --purpose pet-maxpool --batch_size 128 --num_workers 8 --starttrain 0 --endtrain 64000 --startval 64000 --endval 70400 --dropoutrate 0.0 --lr 0.0004 --lr_scheduler_rate 0.99991

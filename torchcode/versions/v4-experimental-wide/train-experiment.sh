#!/bin/bash
python3 main.py --isnewsave --isdebug --gpu 5 --purpose test --batch_size 2 --num_workers 0 --starttrain 0 --endtrain 2 --startval 0 --endval 2 --dropoutrate 0.0 --lr 0.0008 --lr_scheduler_rate 0.99991 --num_thresholds 1

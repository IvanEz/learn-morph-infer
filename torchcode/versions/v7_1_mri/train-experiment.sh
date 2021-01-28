#!/bin/bash
python3 main.py --isnewsave --outputmode 2 --num_epochs 500 --isdebug --gpu 5 --purpose test --batch_size 2 --num_workers 0 --starttrain 0 --endtrain 6 --startval 0 --endval 6 --dropoutrate 0.5 --lr 0.00008 --lr_scheduler_rate 0.99999 --num_thresholds 1

#!/bin/bash

python3 main.py --isnewsave --purpose resnetdeep-64000-2thresholds-schedstepepoch --batch_size 128 --num_workers 16 --starttrain 0 --endtrain 64000 --startval 64000 --endval 70400 --dropoutrate 0.4 --lr 0.0001 --lr_scheduler_rate 0.978 --gpu 5 --num_thresholds 1
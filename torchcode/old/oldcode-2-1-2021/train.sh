#!/bin/bash
python3 main.py --isnewsave --outputmode 3 --gpu 3 --purpose norm-necr-normpet-n4-128 --batch_size 10 --num_workers 5 --starttrain 0 --endtrain 80000 --startval 80000 --endval 88000 --dropoutrate 0.0 --lr 0.00002 --lr_scheduler_rate 0.999998 --weight_decay_sgd 0.01

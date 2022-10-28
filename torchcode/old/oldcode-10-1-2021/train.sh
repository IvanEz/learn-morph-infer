#!/bin/bash
python3 main.py --isnewsave --outputmode 3 --gpu 3 --purpose necr-normpet-normalized --batch_size 25 --num_workers 5 --starttrain 0 --endtrain 80000 --startval 80000 --endval 88000 --dropoutrate 0.0 --lr 0.00008 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.01

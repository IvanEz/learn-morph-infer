#!/bin/bash
python3 main.py --isnewsave --outputmode 3 --gpu 0 --purpose necr-normpet-nonorm-batch32-wd05-xyz --batch_size 32 --num_workers 5 --starttrain 0 --endtrain 80000 --startval 80000 --endval 88000 --dropoutrate 0.0 --lr 0.00006 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.05 --savelogdir="./result/"

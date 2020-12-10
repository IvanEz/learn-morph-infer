#!/bin/sh
#conda activate torchenv
#source setup_ibbm_giga.sh
nohup python3 -u evaluator.py --start 80000 --end 80500 --parapid 0 > output0.txt &
nohup python3 -u evaluator.py --start 80500 --end 81000 --parapid 1 > output1.txt &
nohup python3 -u evaluator.py --start 81000 --end 81500 --parapid 2 > output2.txt &
nohup python3 -u evaluator.py --start 81500 --end 82000 --parapid 3 > output3.txt &
nohup python3 -u evaluator.py --start 82000 --end 82500 --parapid 4 > output4.txt &
nohup python3 -u evaluator.py --start 82500 --end 83000 --parapid 5 > output5.txt &
nohup python3 -u evaluator.py --start 83000 --end 83500 --parapid 6 > output6.txt &
nohup python3 -u evaluator.py --start 83500 --end 84000 --parapid 7 > output7.txt &
nohup python3 -u evaluator.py --start 84000 --end 84500 --parapid 8 > output8.txt &
nohup python3 -u evaluator.py --start 84500 --end 85000 --parapid 9 > output9.txt &
nohup python3 -u evaluator.py --start 85000 --end 85500 --parapid 10 > output10.txt &
nohup python3 -u evaluator.py --start 85500 --end 86000 --parapid 11 > output11.txt &
nohup python3 -u evaluator.py --start 86000 --end 86500 --parapid 12 > output12.txt &
nohup python3 -u evaluator.py --start 86500 --end 87000 --parapid 13 > output13.txt &
nohup python3 -u evaluator.py --start 87000 --end 87500 --parapid 14 > output14.txt &
nohup python3 -u evaluator.py --start 87500 --end 88000 --parapid 15 > output15.txt &
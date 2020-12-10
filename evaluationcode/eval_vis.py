import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import argparse

zerodiceon08 = False
#thresholds = [0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9, 0.95]

parser = argparse.ArgumentParser()
parser.add_argument('--parapros', default=1, type=int) #how many PARAllel PROcesseS (number of resultsX.pkl)
args = parser.parse_args()
print(args)

parapros = args.parapros

allresults = []

for i in range(0, parapros):
    with open('evalresults' + str(i) + '.pkl', 'rb') as evalr:
        evalresults = pickle.load(evalr)
        allresults = allresults + evalresults

## create histograms for dice ##
for i in range(0, len(thresholds)):
    threshold = thresholds[i]
    scores_for_threshold = []
    for result in allresults:
        assert result[2][i][0] == threshold
        dice = result[2][i][1] #dice score for this threshold
        scores_for_threshold.append(dice)

    scoreslen = len(scores_for_threshold)
    nan_scores = np.isnan(np.array(scores_for_threshold)).sum() #count how many nans
    scores_for_threshold = [x for x in scores_for_threshold if not np.isnan(x)] #remove nans for histogram

    f = plt.figure()
    plt.hist(scores_for_threshold, bins=50, range=(0.0, 1.0), figure=f)
    plt.title("DICE >= " + str(threshold) + ", mean = " + str(np.round(np.array(scores_for_threshold).mean(), 3)) + ", nans = " + str(nan_scores) + ", evaluated: " + str(scoreslen))
    plt.savefig('dice_' + str(threshold) + '.png')


if zerodiceon08:
    threshold = thresholds[10]
    tumors = []
    for result in allresults:
        assert result[2][10][0] == threshold
        dice = result[2][10][1]
        if dice == 0.0:
            tumors.append(result)
    for tumor in tumors:
        print(f"{tumor[0]}: {tumor[2]}")
    print(len(tumors))      

##############################
acceptable = 0
ok = 0
good = 0
amazing = 0
for result in allresults:
    val_thresholds = [0.001, 0.25, 0.5, 0.75]
    assert result[2][0][0] == val_thresholds[0]
    assert result[2][6][0] == val_thresholds[1]
    assert result[2][8][0] == val_thresholds[2]
    assert result[2][9][0] == val_thresholds[3]
    
    if result[2][0][1] >= 0.6 and result[2][6][1] >= 0.6 and result[2][8][1] >= 0.6 and (result[2][9][1] >= 0.6 or np.isnan(result[2][9][1])):
        acceptable = acceptable + 1

    if result[2][0][1] >= 0.7 and result[2][6][1] >= 0.7 and result[2][8][1] >= 0.7 and (result[2][9][1] >= 0.7 or np.isnan(result[2][9][1])):
        ok = ok + 1
    
    if result[2][0][1] >= 0.8 and result[2][6][1] >= 0.8 and result[2][8][1] >= 0.8 and (result[2][9][1] >= 0.8 or np.isnan(result[2][9][1])):
        good = good + 1
    
    if result[2][0][1] >= 0.9 and result[2][6][1] >= 0.9 and result[2][8][1] >= 0.9 and (result[2][9][1] >= 0.9 or np.isnan(result[2][9][1])):
        amazing = amazing + 1

print(f"acceptable: {acceptable} ({(acceptable / scoreslen)*100}%), ok: {ok} ({(ok / scoreslen)*100}%), good: {good} ({(good / scoreslen)*100}%), amazing: {amazing} ({(amazing / scoreslen)*100}%)")
    
'''
    f = plt.figure()
    plt.hist(scores_for_threshold, bins=100, range=(0.0, 1.0), figure=f)
    plt.title("DICE >= " + str(threshold) + ", mean = " + str(np.round(np.array(scores_for_threshold).mean(), 3)))
    plt.savefig('dice_' + str(threshold) + '.png')
'''

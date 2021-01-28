import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import argparse
from glob import glob

def plothist(data, name):
    f = plt.figure()
    plt.hist(data, bins=50, figure=f)
    plt.title(name + ", mean = " + str(
        np.round(np.array(data).mean(), 3)) + ", evaluated: " + str(len(data)))
    plt.savefig(name + '.png')
    plt.close(f)

thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9, 0.95]

parser = argparse.ArgumentParser()
parser.add_argument('--parapros', default=1, type=int)  # how many PARAllel PROcesseS (number of evaluation-*.pkl)
parser.add_argument('--start', default=0, type=int) #inclusive
parser.add_argument('--stop', default=1, type=int) #exclusive
parser.add_argument('--datadir', default="/mnt/Drive2/ivan_kevin/testsetdata", type=str)
args = parser.parse_args()
print(args)

parapros = args.parapros
datadir = args.datadir
start = args.start
stop = args.stop

allresults = []

for i in range(0, parapros):
    with open('evaluation' + str(i) + '.pkl', 'rb') as evalr:
        evalresults = pickle.load(evalr)
        print(len(evalresults))
        allresults = allresults + evalresults

print(len(allresults))

paths = []

for result in allresults:
    paths.append(result['path'])

print(len(paths))
toevaluatepaths = sorted(glob("{}/*/".format(datadir)))[start : stop]
print(len(toevaluatepaths))
assert sorted(paths) == toevaluatepaths
print("all specified data has been processed - no duplicates or missing")


## create histograms for dice ##
for i in range(0, len(thresholds)):
    threshold = thresholds[i]
    scores_for_threshold = []
    for result in allresults:
        assert result['diceresults'][i][0] == threshold
        dice = result['diceresults'][i][1]  # dice score for this threshold
        scores_for_threshold.append(dice)

    scoreslen = len(scores_for_threshold)
    nan_scores = np.isnan(np.array(scores_for_threshold)).sum()  # count how many nans
    scores_for_threshold = [x for x in scores_for_threshold if not np.isnan(x)]  # remove nans for histogram

    f = plt.figure()
    plt.hist(scores_for_threshold, bins=50, range=(0.0, 1.0), figure=f)
    plt.title("DICE >= " + str(threshold) + ", mean = " + str(
        np.round(np.array(scores_for_threshold).mean(), 3)) + ", nans = " + str(nan_scores) + ", evaluated: " + str(
        scoreslen))
    plt.savefig('dice_' + str(threshold) + '.png')
    plt.close(f)

##############################
acceptable = 0
ok = 0
good = 0
amazing = 0
for result in allresults:
    val_thresholds = [0.001, 0.25, 0.5, 0.75]
    assert result['diceresults'][0][0] == val_thresholds[0]
    assert result['diceresults'][6][0] == val_thresholds[1]
    assert result['diceresults'][8][0] == val_thresholds[2]
    assert result['diceresults'][9][0] == val_thresholds[3]

    if (result['diceresults'][0][1] >= 0.6 or np.isnan(result['diceresults'][0][1])) and (
            result['diceresults'][6][1] >= 0.6 or np.isnan(result['diceresults'][6][1])) and (
            result['diceresults'][8][1] >= 0.6 or np.isnan(result['diceresults'][8][1])) and (
            result['diceresults'][9][1] >= 0.6 or np.isnan(result['diceresults'][9][1])):
        acceptable = acceptable + 1

    if (result['diceresults'][0][1] >= 0.7 or np.isnan(result['diceresults'][0][1])) and (
            result['diceresults'][6][1] >= 0.7 or np.isnan(result['diceresults'][6][1])) and (
            result['diceresults'][8][1] >= 0.7 or np.isnan(result['diceresults'][8][1])) and (
            result['diceresults'][9][1] >= 0.7 or np.isnan(result['diceresults'][9][1])):
        ok = ok + 1

    if (result['diceresults'][0][1] >= 0.8 or np.isnan(result['diceresults'][0][1])) and (
            result['diceresults'][6][1] >= 0.8 or np.isnan(result['diceresults'][6][1])) and (
            result['diceresults'][8][1] >= 0.8 or np.isnan(result['diceresults'][8][1])) and (
            result['diceresults'][9][1] >= 0.8 or np.isnan(result['diceresults'][9][1])):
        good = good + 1

    if (result['diceresults'][0][1] >= 0.9 or np.isnan(result['diceresults'][0][1])) and (
            result['diceresults'][6][1] >= 0.9 or np.isnan(result['diceresults'][6][1])) and (
            result['diceresults'][8][1] >= 0.9 or np.isnan(result['diceresults'][8][1])) and (
            result['diceresults'][9][1] >= 0.9 or np.isnan(result['diceresults'][9][1])):
        amazing = amazing + 1

print(
    f"acceptable: {acceptable} ({(acceptable / scoreslen) * 100}%), ok: {ok} ({(ok / scoreslen) * 100}%), good: {good} ({(good / scoreslen) * 100}%), amazing: {amazing} ({(amazing / scoreslen) * 100}%)")



ic_absolute_errors = []
#icx_absolute_errors = []
#icy_absolute_errors = []
#icz_absolute_errors = []
#mu1_relative_errors = []
#mu2_relative_errors = []

for result in allresults:
    icx_gt = result['icx_gt']
    icy_gt = result['icy_gt']
    icz_gt = result['icz_gt']
    icx_predicted = result['icx_predicted']
    icy_predicted = result['icy_predicted']
    icz_predicted = result['icz_predicted']
    #mu1_gt = result['mu1_gt']
    #mu2_gt = result['mu2_gt']
    #mu1_predicted = result['mu1_predicted']
    #mu2_predicted = result['mu2_predicted']

    #mu1_absolute_error = np.abs(mu1_predicted - mu1_gt)
    #mu2_absolute_error = np.abs(mu2_predicted - mu2_gt)
    icx_absolute_error = np.abs(icx_predicted - icx_gt)
    icy_absolute_error = np.abs(icy_predicted - icy_gt)
    icz_absolute_error = np.abs(icz_predicted - icz_gt)

    ic_absolute_error = (icx_absolute_error + icy_absolute_error + icz_absolute_error) / 3
    ic_absolute_errors.append(ic_absolute_error)

    #mu1_relative_error = np.abs(1 - (mu1_predicted / mu1_gt))
    #mu2_relative_error = np.abs(1 - (mu2_predicted / mu2_gt))

    #icx_absolute_errors.append(icx_absolute_error)
    #icy_absolute_errors.append(icy_absolute_error)
    #icz_absolute_errors.append(icz_absolute_error)

    #mu1_relative_errors.append(mu1_relative_error)
    #mu2_relative_errors.append(mu2_relative_error)


'''
f = plt.figure()
plt.hist(icx_absolute_errors, bins=50, figure=f)
plt.title("icx_absolute_errors" + ", mean = " + str(np.round(np.array(icx_absolute_errors).mean(), 3)) + ", evaluated: " + str(len(icx_absolute_errors)))
plt.savefig('icx_absolute_errors.png')
'''

plothist(ic_absolute_errors, "ic_absolute_errors")
#plothist(icx_absolute_errors, "icx_absolute_errors")
#plothist(icy_absolute_errors, "icy_absolute_errors")
#plothist(icz_absolute_errors, "icz_absolute_errors")
#plothist(mu1_relative_errors, "mu1_relative_errors")
#plothist(mu2_relative_errors, "mu2_relative_errors")
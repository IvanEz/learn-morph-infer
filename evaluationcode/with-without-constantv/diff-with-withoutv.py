import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt

parapros = 16

allresults0 = []
allresults1 = []

dice_differences = []

for i in range(0, parapros):
    with open('noconstantv/evalresults' + str(i) + '.pkl', 'rb') as evalr0:
            evalresults = pickle.load(evalr0)
            allresults0 = allresults0 + evalresults

    with open('constantv/evalresults' + str(i) + '.pkl', 'rb') as evalr1:
            evalresults = pickle.load(evalr1)
            allresults1 = allresults1 + evalresults

assert len(allresults0) == len(allresults1)
print(len(allresults0))

for i in range(0, len(allresults0)): #tumor i
    assert allresults0[i][0] == allresults1[i][0]
    assert allresults0[i][1] == allresults1[i][1]
    assert len(allresults0[i][2]) == len(allresults1[i][2])

    for j in range(0, len(allresults0[i][2])): #threshold j
        assert allresults0[i][2][j][0] == allresults1[i][2][j][0]

        dice_diff = np.abs(allresults0[i][2][j][1] - allresults1[i][2][j][1])
        if dice_diff > 0.0:
            dice_differences.append(dice_diff)

print(f"max diff: {np.max(dice_differences)}")
plt.hist(dice_differences, bins=50)
plt.show()

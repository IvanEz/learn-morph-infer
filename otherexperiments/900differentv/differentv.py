import numpy as np
import mayavi.mlab as mlab
import pickle5 as pickle
import subprocess
import matplotlib.pyplot as plt
import random
from glob import glob
#difference for 3766, 47209: < 10^-6
#chosen = 34567 #3766

paths = sorted(glob("{}/*/".format("/mnt/Drive2/ivan_kevin/samples_extended_copy/Dataset/")))[1018:2000]

for x in paths:
    f = open("differentvresults.txt", "a")
    chosen = str(x)
    print(x)

    path = x

    with open(path + "parameter_tag.pkl", "rb") as params:
        params = pickle.load(params)
        print("Selected tumor has parameter tag: " + str(params))
        D = params['Dw'] * 100 * 365 #mm² / yr
        rho = params['rho'] * 365 #1 / yr
        T = params['Tend'] / 365  #yr

        alpha = D / rho
        beta = T * rho

        icx = params['icx']
        icy = params['icy']
        icz = params['icz']

        #icx = 0.0
        #icy = 0.0
        #icz = 0.0

        v = 2 * np.sqrt(D * rho) #mm / yr
        print(f"Selected tumor has Dw = {D} mm^2/yr, p = {rho} 1/yr, T = {T} yr, D/p = {alpha} mm^2, Tp = {beta}, v = {v} mm/yr, icx = {icx}, icy = {icy}, icz = {icz} \n")

        if v < 100.0:
            v2 = 100 * np.random.rand() + 100 #new velocity
        else:
            v2 = 80 * np.random.rand() + 20
        
        #v2 = 43.05036445126889 #debug

        #print(v2)
        D2 = (v2 / 2) * np.sqrt(alpha)
        p2 = (v2 / 2) / np.sqrt(alpha)
        T2 = beta / p2

        alpha2 = D2 / p2 #should be equal to alpha
        beta2 = T2 * p2

        print(f"New tumor with different velocity: Dw = {D2} mm^2/yr, p = {p2} 1/yr, T = {T2} yr, D/p = {alpha2} mm^2, Tp = {beta2}, v = {v2} mm/yr, icx = {icx}, icy = {icy}, icz = {icz} \n")

        D2 = D2 * (1 / 100) * (1 / 365) #in cm^2/d
        p2 = p2 * (1 / 365) #in 1/yr
        T2 = T2 * 365 

        command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(D2) + " -rho " + str(p2) + " -Tend " + str(T2) + " -dumpfreq " + str(0.9999 * T2) + " -icx " + str(icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0" #-bDumpIC 1

        print(f"Inputting D2 = {D2} cm^2/d, p2 = {p2} 1/yr, T2 = {T2}d with command {command}")

        simulation = subprocess.check_call([command], shell=True, cwd="./vtus/sim/")
        vtu2npz = subprocess.check_call(["python3 vtutonpz.py"], shell = True)

        tumor1 = np.load(path + "Data_0001.npz")['data']
        tumor2 = np.load("./npzs/sim/Data_0001.npz")['data'][:,:,:,0]

        diff = np.abs(tumor2 - tumor1)
        
        diff = (diff >= 0.0001) * diff #threshold at 0.01% and retain values bigger than 0.01%
        #diff = (diff >= 0.00001) * diff #for debug
        nonzerodiff = np.count_nonzero(diff)
        print("voxel difference: " + str(nonzerodiff))
        f.write(str(x) + ": vold = " + str(v) + ", vnew = " + str(v2) + ", difference = " + str(nonzerodiff) + ", maxdiff = " + str(np.amax(diff)) + "\n")
        f.close()

import numpy as np
import mayavi.mlab as mlab
import pickle5 as pickle
import subprocess
import matplotlib.pyplot as plt
import random
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sqrtDT', default=7.0, type=float)
parser.add_argument('--sqrtTp', default=3.5, type=float)
parser.add_argument('--v', default=150.0, type=float)
args = parser.parse_args()
print(args)

icx = 0.39078
icy = 0.4081
icz = 0.63007

#difference for 3766, 47209: < 10^-6
#chosen = 34567 #3766
D_interval = np.array([0.0002, 0.015]) * 100 * 365 #mm²/yr
p_interval = np.array([2/365, 0.2]) * 365 #1/yr
T_interval = np.array([50, 1500]) / 365 #yr

sqrtdt_grid = np.array([args.sqrtDT]) #mm
v_grid = np.array([args.v]) #mm/yr
mu_grid = np.array([args.sqrtTp]) #constant (sqrt(Tp))

Ds = []
ps = []
Ts = []
name = []

for x in mu_grid:
    for y in sqrtdt_grid:
        for z in v_grid:
            lamb = y / x
            D = (lamb * z) / 2 #mm^2 / yr
            p = z / (2*lamb) #1 / yr
            #T = x / p #yr
            T = (2*y*x)/z

            Ds.append(D)
            ps.append(p)
            Ts.append(T)
            name.append(str(y) + "-" + str(z) + "-" + str(x))
            

print(Ds)
print(ps)
print(Ts)
print(name)


for tumor in enumerate(name):
    #D2 = (v2 / 2) * np.sqrt(alpha)
    #p2 = (v2 / 2) / np.sqrt(alpha)
    #T2 = beta / p2
    print(tumor[1])
    D2 = Ds[tumor[0]]
    p2 = ps[tumor[0]]
    T2 = Ts[tumor[0]]

    alpha2 = D2 / p2 #should be equal to alpha
    beta2 = T2 * p2
    gamma2 = np.sqrt(alpha2 * beta2)
    v2 = 2 * np.sqrt(D2 * p2)

    #assert alpha2 == alpha
    #print(f"{alpha2} == {alpha}")
    #print(f"{beta2} == {beta}")
    #assert beta2 == beta

    print(f"Tumor: Dw = {D2} mm^2/yr, p = {p2} 1/yr, T = {T2} yr, D/p = {alpha2} mm^2, Tp = {beta2}, sqrt(DT) = {gamma2}, sqrt(Tp) = {np.sqrt(beta2)} v = {v2} mm/yr, icx = {icx}, icy = {icy}, icz = {icz} \n")

    D2 = D2 * (1 / 100) * (1 / 365) #in cm^2/d
    p2 = p2 * (1 / 365) #in 1/yr
    T2 = T2 * 365 

    if (D2 >= 0.0002 and D2 <= 0.015) and (p2 >= 0.002 and p2 <= 0.2) and (T2 >= 50 and T2 <= 1500):
        print("parameter set in range")
    else:
        print("PARAMETER SET OUT OF RANGE!")

    #command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(D2) + " -rho " + str(p2) + " -Tend " + str(T2) + " -dumpfreq " + str(T2) + " -icx " + str(icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0" #-bDumpIC 1

    command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(D2) + " -rho " + str(p2) + " -Tend " + str(T2) + " -dumpfreq " + str(0.999 * T2) + " -icx " + str(icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0" #-bDumpIC 1

    print(f"Inputting D2 = {D2} cm^2/d, p2 = {p2} 1/yr, T2 = {T2}d with command {command}")

    simulation = subprocess.check_call([command], shell=True, cwd="./vtus/sim/")
    vtu2npz = subprocess.check_call(["python3 vtutonpz.py --name " + tumor[1] + "-"], shell = True)


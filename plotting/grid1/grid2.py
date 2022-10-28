import numpy as np
import mayavi.mlab as mlab
import pickle5 as pickle
import subprocess
import matplotlib.pyplot as plt
import random
from glob import glob

icx = 0.39078
icy = 0.4081
icz = 0.63007

#difference for 3766, 47209: < 10^-6
#chosen = 34567 #3766
D_interval = np.array([0.0002, 0.015]) * 100 * 365 #mm²/yr
p_interval = np.array([2/365, 0.2]) * 365 #1/yr
T_interval = np.array([50, 1500]) / 365 #yr
'''
def generate_log_grid(interval, gridpoints):
    interval = np.log10(interval)
    spacing = (interval[1] - interval[0]) / gridpoints

    points = np.empty(gridpoints)
    points[0] = interval[0] + spacing / 2
    for i in range(1, gridpoints):
        points[i] = points[i-1] + spacing

    return points

D_interval = np.array([0.0002, 0.015]) * 100 * 365 #mm²/yr
p_interval = np.array([2/365, 0.2]) * 365 #1/yr
T_interval = np.array([50, 1500]) / 365 #yr

lambda_interval = np.array([np.sqrt(2), np.sqrt(20)])
v_interval = np.array([10, 200])
mu_2_grid = np.array([6, 10, 12])
gridpoints = 4

####
generate_log_grid(D_interval, 4)


D_loggrid = generate_log_grid(D_interval, 4)
p_loggrid = generate_log_grid(p_interval, 4)
print(D_loggrid)
print(p_loggrid)

D_grid = 10**D_loggrid
p_grid = 10**p_loggrid
print(D_grid)
print(p_grid)
'''
'''
D_interval = np.log10(D_interval)
D_spacing = (D_interval[1] - D_interval[0]) / gridpoints

D_points = np.empty(gridpoints)
D_points[0] = D_interval[0] + D_spacing / 2
for i in range(1, gridpoints):
    D_points[i] = D_points[i-1] + D_spacing

print(D_points)
'''

lambda_grid = np.sqrt(np.array([3,9.5,16])) #mm
v_grid = np.array([50,300]) #mm/yr
mu_grid = np.array([10,15,20]) #constant

Ds = []
ps = []
Ts = []
name = []

for x in mu_grid:
    for y in lambda_grid:
        for z in v_grid:
            D = (y * z) / 2 #mm^2 / yr
            p = z / (2*y) #1 / yr
            T = x / p #yr

            Ds.append(D)
            ps.append(p)
            Ts.append(T)
            name.append(str(int(np.round(y**2,5))) + "-" + str(z) + "-" + str(x))
            #format is lambda-v-mu

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

    print(f"Tumor: Dw = {D2} mm^2/yr, p = {p2} 1/yr, T = {T2} yr, D/p = {alpha2} mm^2, Tp = {beta2}, sqrt(DT) = {gamma2}, v = {v2} mm/yr, icx = {icx}, icy = {icy}, icz = {icz} \n")

    D2 = D2 * (1 / 100) * (1 / 365) #in cm^2/d
    p2 = p2 * (1 / 365) #in 1/yr
    T2 = T2 * 365 

    #command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(D2) + " -rho " + str(p2) + " -Tend " + str(T2) + " -dumpfreq " + str(T2) + " -icx " + str(icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0" #-bDumpIC 1

    command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(D2) + " -rho " + str(p2) + " -Tend " + str(T2) + " -dumpfreq " + str(0.999 * T2) + " -icx " + str(icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0" #-bDumpIC 1

    print(f"Inputting D2 = {D2} cm^2/d, p2 = {p2} 1/yr, T2 = {T2}d with command {command}")

    simulation = subprocess.check_call([command], shell=True, cwd="./vtus/sim/")
    vtu2npz = subprocess.check_call(["python3 vtutonpz.py --name " + tumor[1] + "-"], shell = True)


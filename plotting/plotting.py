import matplotlib.pyplot as plt
import matplotlib
import pickle5 as pickle
from glob import glob
import numpy as np
import mayavi.mlab as mayaplot

rootdir = "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/"
#rootdir = "/mnt/Drive2/ivan_kevin/samples_extended_copy_testset/"
paths = sorted(glob("{}/*/".format(rootdir)))[0:80000] #0:80000


def v(diff, prol):
    return 2 * np.sqrt(diff * prol)

def v_10(diff):
    return 25 / diff

def v_200(diff):
    return 10000 / diff

def Ddivp2(diff):
    return diff / 2

def Ddivp20(diff):
    return diff / 20

def plot_D_p():
    D = []
    rho = []
    for path in paths:
        with open(path + "parameter_tag2.pkl", "rb") as params:
            params = pickle.load(params)
            D_current = params['Dw'] * 100 * 365
            rho_current = params['rho'] * 365
            D.append(D_current)
            rho.append(rho_current)
            

    vfunc = np.vectorize(v)
    velocities = vfunc(D, rho)
    print(f"Min velocity: {np.min(velocities)}")
    v10func = np.vectorize(v_10)
    v200func = np.vectorize(v_200)
    ddivp2func = np.vectorize(Ddivp2)
    ddivp20func = np.vectorize(Ddivp20)

    D_range = np.linspace(5, 548)

    fig, ax = plt.subplots()
    #im = ax.scatter(Ds, ps, c = 2 * np.sqrt(Ds * ps), s=30, cmap='viridis', edgecolors='red')
    im = ax.scatter(D, rho,s=1,c='darkgray')
    #im = ax.scatter(jana_dw, jana_p, c = 2 * np.sqrt(jana_dw * jana_p), s=30, cmap='viridis', edgecolors='red')
    ##plt.axline((10,2.5), (25, 1), linestyle='--', linewidth=0.8)
    ##plt.axline((1,10000), (2, 5000), linestyle='--', linewidth=0.8)
    ##plt.axline((1, 0.5), (2, 1.0), linestyle='--', linewidth=0.8)
    ##plt.axline((20,1), (200, 10), linestyle='--', linewidth=0.8)


    plt.plot(D_range, v10func(D_range), c='black', linestyle='dashed')
    plt.plot(D_range, v200func(D_range), c='black', linestyle='dashed')
    plt.plot(D_range, ddivp2func(D_range), c='black', linestyle='dashed')
    plt.plot(D_range, ddivp20func(D_range), c='black', linestyle='dashed')
    ##plt.axline((10,1000), (500, 20))
    #clb = fig.colorbar(im, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ##plt.axline((7.3, 0.5), (2, 1.0), linestyle='--', linewidth=0.8)
    #plt.axvline(x=7.3, c='black')
    #plt.axvline(x=547.5, c='black')

    #plt.axhline(y=0.73, c='black')
    #plt.axhline(y=73.0, c='black')

    plt.xlabel("D in mm²/yr")
    plt.ylabel("rho in 1/yr")

    plt.xlim(5, 800)
    plt.ylim(0.8, 100)
    #clb.ax.set_xlabel("v in mm/yr")
    plt.show()
############################################################################################################
def plot_sqrtDrho_sqrtTp_v_experiment():
    x = [] #D/rho
    y = [] #T * p
    color = []

    left_x = []
    left_y = []
    left_color = []
    mid_x = []
    mid_y = []
    mid_color = []
    right_x = []
    right_y = []
    right_color = []


    for path in paths:
        with open(path + "parameter_tag2.pkl", "rb") as params:
            params = pickle.load(params)
            D_current = params['Dw'] * 100 * 365 #mm^2/yr
            rho_current = params['rho'] * 365 #1/yr
            T_current = params['Tend'] / 365 #yr
            v_current = 2 * np.sqrt(D_current * rho_current) #mm/yr
            x.append(np.sqrt(D_current / rho_current)) #sqrt(D/p) (mm)
            y.append(np.sqrt(T_current * rho_current)) #Tp (constant)
            color.append(v_current) #v = 2 * sqrt(D * rho)
            '''
            if (np.round(np.sqrt(D_current / rho_current),1) == 0.7 and np.round(np.sqrt(T_current * rho_current),1) == 7):
                print(f"path for left: {path}")
            elif (np.round(np.sqrt(D_current / rho_current),2) == 2.1 and np.round(np.sqrt(T_current * rho_current),2) == 4.2):
                print(f"path for mid: {path}")
            elif (np.round(np.sqrt(D_current / rho_current),1) == 7 and np.round(np.sqrt(T_current * rho_current),2) == 3.5):
                print(f"path for right: {path}")
            '''

            if path == "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/42483/":
                left_x.append(np.sqrt(D_current / rho_current)) #sqrt(D/p) (mm)
                left_y.append(np.sqrt(T_current * rho_current)) #Tp (constant)
                left_color.append(v_current) #v = 2 * sqrt(D * rho)

            if path == "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/36706/":
                mid_x.append(np.sqrt(D_current / rho_current)) #sqrt(D/p) (mm)
                mid_y.append(np.sqrt(T_current * rho_current)) #Tp (constant)
                mid_color.append(v_current) #v = 2 * sqrt(D * rho)

            if path == "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/66504/":
                right_x.append(np.sqrt(D_current / rho_current)) #sqrt(D/p) (mm)
                right_y.append(np.sqrt(T_current * rho_current)) #Tp (constant)
                right_color.append(v_current) #v = 2 * sqrt(D * rho)

    print("len: " + str(len(x)))



    #D_range = np.linspace(7.3, 548)
    #jana_dw = np.array([0.188, 0.06, 0.846, 0.223, 2.447, 0.44, 0.196, 0.484]) #mm²/d
    #jana_p = np.array([0.029, 0.024, 0.01, 0.01, 0.107, 0.029, 0.007, 0.025]) #1/d
    #jana_T = np.array([273.130, 337.582, 1094.225, 848.5, 93.278, 406.237, 1580.353, 391.577]) #d

    fig, ax = plt.subplots()
    #im = ax.scatter(x, y, c = color, s=10, cmap='viridis')

    im = ax.scatter(x, y, c = color, s=10, cmap='viridis', vmin=4, vmax=400)

    im = ax.scatter(left_x, left_y, c = left_color, s=60, cmap='viridis', edgecolors = 'red', linewidths=2, vmin=4, vmax=400)
    
    im = ax.scatter(mid_x, mid_y, c = mid_color, s=60, cmap='viridis', edgecolors = 'lightgrey', linewidths=2, vmin=4, vmax=400)
    
    im = ax.scatter(right_x, right_y, c = right_color, s=60, cmap='viridis', edgecolors = 'cyan', linewidths=2, vmin=4, vmax=400)
    
    #im = ax.scatter(np.sqrt(jana_dw / jana_p), jana_T * jana_p, c = 2 * np.sqrt(jana_dw * jana_p) * 365, s=50, cmap='viridis', edgecolors='red')
    #plt.plot(D_range, v10func(D_range), c='black')
    #plt.plot(D_range, v100func(D_range), c='black')
    #plt.plot(D_range, ddivp2func(D_range), c='black')
    #plt.plot(D_range, ddivp20func(D_range), c='black')
    clb = fig.colorbar(im, ax=ax)
    #plt.axline((2,0), (2, 1), linestyle='--', linewidth=0.8)
    #plt.axline((20,0), (20, 1), linestyle='--', linewidth=0.8)
    plt.axvline(x=np.sqrt(2), c='black')
    plt.axvline(x=np.sqrt(20), c='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel("sqrt(D/p) in mm")
    plt.ylabel("sqrt(Tp) as constant")
    clb.ax.set_xlabel("v in mm/yr")

    axins = ax.inset_axes([0.73, 0.73, 0.25, 0.25])
    axins.scatter(x, y, c = color, s=10, cmap='viridis', vmin=4, vmax=400)
    # sub region of the original image
    axins.scatter(mid_x, mid_y, c = mid_color, s=60, cmap='viridis', edgecolors = 'lightgrey', linewidths=2, vmin=4, vmax=400)
    x1, x2, y1, y2 = 2, 3.2, 3.75, 4.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.set_xscale('log')
    ax.set_yscale('log')
    

    ax.indicate_inset_zoom(axins)



    plt.show()



plot_D_p()
#plot_sqrtDrho_sqrtTp_v_experiment()


#https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/zoom_inset_axes.html
#https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
#https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot

import numpy as np
import argparse
from glob import glob
import pickle5 as pickle
import subprocess
import time

# from mayavi import mlab

##################################
#
# DON'T FORGET TO CALL "source ./setup_ibbm_giga.sh" so libraries work!
#
##################################

normalization_range = [-1.0, 1.0]
# thresholds = [0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9, 0.95]

def assertrange(Dw,rho,Tend):
    if not ((Dw >= 0.0002 and Dw <= 0.015) and (rho >= 0.002 and rho <= 0.2) and (Tend >= 50 and Tend <= 1500)):
        print("LIGHT WARNING: parameter(s) out of generated range")

# if one of the predicted parameters is out of range, numpy interpolate automatically brings it to border
def convert(l, m, v, x, y, z):
    l = np.interp(l, normalization_range, [np.sqrt(0.001), np.sqrt(7.5)])  # cm
    m = np.interp(m, normalization_range, [0.1, 300.0])  # constant
    v = np.interp(v, normalization_range, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)])  # cm / d
    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = (l * v) / 2.0  # cm^2 / d
    p = v / (2.0 * l)  # 1 / d
    T = m / p  # days

    return (
    D.astype(np.float32), p.astype(np.float32), T.astype(np.float32), x.astype(np.float32), y.astype(np.float32),
    z.astype(np.float32))


# should i cite https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py? Looked up how to calc dice although it was clear from definition, but wanted to be sure
def dice(gt, sim):
    result = []
    for threshold in thresholds:
        gt_thresholded = (gt >= threshold)
        sim_thresholded = (sim >= threshold)

        total = gt_thresholded.sum() + sim_thresholded.sum()
        # assert total != 0

        intersect = (gt_thresholded & sim_thresholded).sum()
        if total != 0:
            result.append((threshold, ((2 * intersect) / total)))
        else:
            result.append((threshold, np.nan))

    return result


parser = argparse.ArgumentParser()
parser.add_argument('--start', default=80000, type=int)
parser.add_argument('--end', default=88000, type=int)
parser.add_argument('--parapid', default=0, type=int)
parser.add_argument('--isdebug', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--constantv', action='store_true')
parser.add_argument('--randomv', action='store_true')
parser.add_argument("--seed", default=123, type=int)
args = parser.parse_args()

print(args)
start = args.start
end = args.end
parapid = args.parapid
isdebug = args.isdebug
visualize = args.visualize
constantv = args.constantv
randomv = args.randomv
seed = args.seed
print(f"start: {start}, end: {end}")
diceresults = []

if visualize:
    print("WARNING: visualization mode on")
    from mayavi import mlab

    time.sleep(8)

if constantv:
    print("WARNING: will use fixed constant velocity to infer parameters!")
    time.sleep(15)

if randomv:
    print("WARNING: will use random velocity to infer parameters")
    npseed = seed + parapid
    np.random.seed(npseed)
    print(f"Seeded random num generator with {npseed}")
    time.sleep(15)

assert not (constantv and randomv) #need to choose between the two

#TODO: currently only works with val set, code needs to be modified to work with separate test set!
with np.load("results.npz") as results:
    start_index = start - 80000
    end_index = end - 80000
    assert start_index >= 0 and start_index <= 20000
    assert end_index >= 0 and end_index <= 20000
    assert start_index < end_index

    ys = results['ys'][start_index : end_index, :]
    yspredicted = results['yspredicted'][start_index : end_index, :]

datapath = "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/"
thrpath = "/mnt/Drive2/ivan_kevin/thresholds/files"

all_paths = sorted(glob("{}/*/".format(datapath)))[start : end]
thr_paths = sorted(glob("{}/*".format(thrpath)))[start : end]

assert len(all_paths) == len(thr_paths)
print(len(all_paths))

for i in range(0, len(all_paths)):
    print(i)
    path = all_paths[i]
    y = ys[i]
    ypredicted = yspredicted[i]

    print(f"path: {path}")
    print(f"y: {y}")
    print(f"ypredicted: {ypredicted}")

    with open(path + "parameter_tag2.pkl", "rb") as par:
        params = pickle.load(par)

    gt_inrange_inpkl = (params['Dw'], params['rho'], params['Tend'], params['icx'], params['icy'], params['icz'])
    gt_inrange_innpz = convert(y[2], y[3], y[4], y[5], y[6], y[7])

    if not isdebug:
        if not constantv:
            if not randomv:
                predicted_inrange = convert(ypredicted[2], ypredicted[3], ypredicted[4], ypredicted[5], ypredicted[6],
                                            ypredicted[7])
            else: #randomv
                v_random = np.interp(np.random.rand(), [0.0, 1.0], normalization_range)
                predicted_inrange = convert(ypredicted[2], ypredicted[3], v_random, ypredicted[5], ypredicted[6],
                                            ypredicted[7])
                print(f"v_random: {v_random}")

        else: #constantv
            predicted_inrange = convert(ypredicted[2], ypredicted[3], 0.0, ypredicted[5], ypredicted[6],
                                        ypredicted[7])
    else:
        print("WARNING: you are in debug mode! RESULTS WILL HAVE PERFECT SCORE!")
        time.sleep(15)
        predicted_inrange = gt_inrange_innpz

    print(f"GROUND TRUTH: {gt_inrange_inpkl}")
    print(f"GROUND TRUTH in results.npz: {gt_inrange_innpz}")
    print(f"PREDICTED: {predicted_inrange} with constantv = {constantv}")

    Dw = predicted_inrange[0]
    rho = predicted_inrange[1]
    Tend = predicted_inrange[2]
    icx = predicted_inrange[3]
    icy = predicted_inrange[4]
    icz = predicted_inrange[5]

    assertrange(Dw,rho,Tend)

    command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(
        Dw) + " -rho " + str(rho) + " -Tend " + str(Tend) + " -dumpfreq " + str(0.9999 * Tend) + " -icx " + str(
        icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0"  # -bDumpIC 1

    print(f"command: {command}")

    simulation = subprocess.check_call([command], shell=True,
                                       cwd="./vtus" + str(parapid) + "/sim/")  # e.g. ./vtus0/sim/
    vtu2npz = subprocess.check_call(["python3 vtutonpz2.py --parapid " + str(parapid)], shell=True)

    with np.load(path + "/Data_0001_thr2.npz") as gttumor:
        gt_tumor = gttumor['data']

    with np.load("npzs" + str(parapid) + "/sim/Data_0001.npz") as simtumor:
        sim_tumor = simtumor['data'][:, :, :, 0]
        sim_tumor_brain = simtumor['data'][:, :, :, 1]
    # gt_tumor = np.load(path + "/Data_0001_thr2.npz")['data']
    # sim_tumor = np.load("npzs" + str(parapid) + "/sim/Data_0001.npz")['data'][:,:,:,0]

    dicescores = dice(gt_tumor, sim_tumor)
    print(dicescores)

    if visualize:
        thr_path = thr_paths[i]
        print(f"thr_path: {thr_path}")
        with np.load(thr_path) as thresholdsfile:
            t1gd_thr = thresholdsfile['t1gd'][0]
            flair_thr = thresholdsfile['flair'][0]

        print(f"t1gd_thr: {t1gd_thr}, flair_thr: {flair_thr}")
        t1gd_volume = (gt_tumor >= t1gd_thr).astype(float)
        flair_volume = (gt_tumor >= flair_thr).astype(float)
        thresholded_volume = 0.666 * t1gd_volume + 0.333 * flair_volume
        pet_volume = ((gt_tumor >= t1gd_thr) * gt_tumor) / 0.5

        f = mlab.figure()
        mlab.volume_slice(gt_tumor, figure=f)
        mlab.title("gt", figure=f)
        f = mlab.figure()
        mlab.volume_slice(sim_tumor, figure=f)
        mlab.title("sim", figure=f)
        f = mlab.figure()
        mlab.volume_slice(sim_tumor_brain, figure=f)
        mlab.title("sim_brain", figure=f)
        f = mlab.figure()
        mlab.volume_slice(thresholded_volume, figure=f)
        mlab.title("thr_volume", figure=f)
        f = mlab.figure()
        mlab.volume_slice(pet_volume, figure=f)
        mlab.title("pet_volume", figure=f)
        mlab.show()

    diceresults.append([(start + i), gt_inrange_inpkl, dicescores])

with open('evalresults' + str(parapid) + '.pkl', 'wb') as save:
    pickle.dump(diceresults, save)

print(args)
print("FINISHED")

'''
import numpy as np
import argparse
from glob import glob
import pickle5 as pickle
import subprocess
import time
from mayavi import mlab

##################################
#
# DON'T FORGET TO CALL "source ./setup_ibbm_giga.sh" so libraries work!
#
##################################

normalization_range = [-1.0, 1.0]
#thresholds = [0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8]

#if one of the predicted parameters is out of range, numpy interpolate automatically brings it to border
def convert(l,m,v,x,y,z):
    l = np.interp(l, normalization_range, [np.sqrt(0.001), np.sqrt(7.5)])  #cm
    m = np.interp(m, normalization_range, [0.1, 300.0])  #constant
    v = np.interp(v, normalization_range, [2*np.sqrt(4e-7), 2*np.sqrt(0.003)])  #cm / d
    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = (l * v) / 2.0  #cm^2 / d
    p = v / (2.0 * l) #1 / d
    T = m / p #days

    return (D.astype(np.float32), p.astype(np.float32), T.astype(np.float32), x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))

#should i cite https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py? Looked up how to calc dice although it was clear from definition, but wanted to be sure
def dice(gt, sim):
    result = []
    for threshold in thresholds:
        gt_thresholded = (gt >= threshold)
        sim_thresholded = (sim >= threshold)

        total = gt_thresholded.sum() + sim_thresholded.sum()
        assert total != 0

        intersect = (gt_thresholded & sim_thresholded).sum()
        result.append((threshold, ((2 * intersect) / total)))
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--start', default=80000, type=int)
parser.add_argument('--end', default=88000, type=int)
parser.add_argument('--parapid', default=0, type=int)
parser.add_argument('--isdebug', action='store_true')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

print(args)
start = args.start
end = args.end
parapid = args.parapid
isdebug = args.isdebug
visualize = args.visualize
print(f"start: {start}, end: {end}")
diceresults = []

if visualize:
    print("WARNING: visualization mode on")
    time.sleep(8)

with np.load("results.npz") as results:
    start_index = start - 80000
    end_index = end - 80000
    assert start_index >= 0 and start_index <= 20000
    assert end_index >= 0 and end_index <= 20000
    assert start_index < end_index

    ys = results['ys'][start_index : end_index, :]
    yspredicted = results['yspredicted'][start_index : end_index, :]

datapath = "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/"

all_paths = sorted(glob("{}/*/".format(datapath)))[start : end]

for i in range(0, len(all_paths)):
    path = all_paths[i]
    y = ys[i]
    ypredicted = yspredicted[i]

    print(f"path: {path}")
    print(f"y: {y}")
    print(f"ypredicted: {ypredicted}")

    with open(path + "parameter_tag2.pkl", "rb") as par:
        params = pickle.load(par)

    gt_inrange_inpkl = (params['Dw'], params['rho'], params['Tend'], params['icx'], params['icy'], params['icz'])
    gt_inrange_innpz = convert(y[2], y[3], y[4], y[5], y[6], y[7])

    if not isdebug:
        predicted_inrange = convert(ypredicted[2], ypredicted[3], ypredicted[4], ypredicted[5], ypredicted[6], ypredicted[7])
    else:
        print("WARNING: you are in debug mode! RESULTS WILL HAVE PERFECT SCORE!")
        time.sleep(15)
        predicted_inrange = gt_inrange_innpz

    print(f"GROUND TRUTH: {gt_inrange_inpkl}")
    print(f"GROUND TRUTH in results.npz: {gt_inrange_innpz}")
    print(f"PREDICTED: {predicted_inrange}")

    Dw = predicted_inrange[0]
    rho = predicted_inrange[1]
    Tend = predicted_inrange[2]
    icx = predicted_inrange[3]
    icy = predicted_inrange[4]
    icz = predicted_inrange[5]

    command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(Dw) + " -rho " + str(rho) + " -Tend " + str(Tend) + " -dumpfreq " + str(0.9999 * Tend) + " -icx " + str(icx) + " -icy " + str(icy) + " -icz " + str(icz) + " -vtk 1 -N 16 -adaptive 0" #-bDumpIC 1

    print(f"command: {command}")

    simulation = subprocess.check_call([command], shell=True, cwd="./vtus" + str(parapid) + "/sim/") #e.g. ./vtus0/sim/
    vtu2npz = subprocess.check_call(["python3 vtutonpz2.py --parapid " + str(parapid)], shell = True)

    with np.load(path + "/Data_0001_thr2.npz") as gttumor:
        gt_tumor = gttumor['data']

    with np.load("npzs" + str(parapid) + "/sim/Data_0001.npz") as simtumor:
        sim_tumor = simtumor['data'][:,:,:,0]
        sim_tumor_brain = simtumor['data'][:,:,:,1]
    #gt_tumor = np.load(path + "/Data_0001_thr2.npz")['data']
    #sim_tumor = np.load("npzs" + str(parapid) + "/sim/Data_0001.npz")['data'][:,:,:,0]

    dicescores = dice(gt_tumor, sim_tumor)
    print(dicescores)

    if visualize:
        f = mlab.figure()
        mlab.volume_slice(gt_tumor, figure=f)  
        mlab.title("gt", figure=f)
        f = mlab.figure()
        mlab.volume_slice(sim_tumor, figure=f)  
        mlab.title("sim", figure=f)
        f = mlab.figure()
        mlab.volume_slice(sim_tumor_brain, figure=f)  
        mlab.title("sim_brain", figure=f)
        mlab.show()

    diceresults.append([(start + i), gt_inrange_inpkl, dicescores])

with open('evalresults' + str(parapid) + '.pkl', 'wb') as save:
    pickle.dump(diceresults, save)
'''
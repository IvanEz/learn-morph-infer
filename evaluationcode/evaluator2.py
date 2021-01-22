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

#VERSION 2 OF EVALUATION! Use this when you don't predict (thr1,thr2,lambda,mu,v,x,y,z) but with different output
#modes (version 7) and use necrotic core + normalized pet

normalization_range = [-1.0, 1.0]
# thresholds = [0.001, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9, 0.95]

def assertrange(Dw,rho,Tend):
    if not ((Dw >= 0.0002 and Dw <= 0.015) and (rho >= 0.002 and rho <= 0.2) and (Tend >= 50 and Tend <= 1500)):
        print("LIGHT WARNING: parameter(s) out of generated range")

# if one of the predicted parameters is out of range, numpy interpolate automatically brings it to bound
def convert_3(l, m, x, y, z):
    l = np.interp(l, normalization_range, [np.sqrt(0.001), np.sqrt(7.5)])  # cm
    m = np.interp(m, normalization_range, [0.1, 300.0])  # constant
    v = np.interp(np.mean(normalization_range), normalization_range, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)])  # cm / d
    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = (l * v) / 2.0  # cm^2 / d
    p = v / (2.0 * l)  # 1 / d
    T = m / p  # days

    return (
    D.astype(np.float32), p.astype(np.float32), T.astype(np.float32), x.astype(np.float32), y.astype(np.float32),
    z.astype(np.float32))

def convert_3_reverse(D, p, T, x, y, z):
    return (np.sqrt(D/p), T*p, x,y,z)

def convert_4(sqrtDT, sqrtTp, x, y, z):
    sqrtDT = np.interp(sqrtDT, normalization_range, [0.1, np.sqrt(22.5)])  # cm
    sqrtTp = np.interp(sqrtTp, normalization_range, np.sqrt([0.1, 300.0])) # constant
    v = np.interp(np.mean(normalization_range), normalization_range, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)])  # cm / d
    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = (sqrtDT * v) / (2.0 * sqrtTp)  # cm^2 / d
    p = (sqrtTp * v) / (2.0 * sqrtDT)  # 1 / d
    T = (2.0 * sqrtDT * sqrtTp) / v  # days

    return (
    D.astype(np.float32), p.astype(np.float32), T.astype(np.float32), x.astype(np.float32), y.astype(np.float32),
    z.astype(np.float32))

def convert_4_reverse(D, p, T, x, y, z):
    return (np.sqrt(D*T), np.sqrt(T*p), x,y,z)

def convert_5(D, p, T, x, y, z):
    D = np.interp(D, normalization_range, [0.0002, 0.015])
    p = np.interp(p, normalization_range, [0.002, 0.2])
    T = np.interp(T, normalization_range, [50, 1500])
    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    return (
    D.astype(np.float32), p.astype(np.float32), T.astype(np.float32), x.astype(np.float32), y.astype(np.float32),
    z.astype(np.float32))

def convert_5_reverse(D, p, T, x, y, z): #does nothing, but still here for consistency across different output modes
    return (D,p,T,x,y,z)

#https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py
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
parser.add_argument("--outputmode", default=3, type=int)
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
outputmode = args.outputmode
print(f"start: {start}, end: {end}")
diceresults = []

if visualize:
    print("WARNING: visualization mode on")
    from mayavi import mlab

    time.sleep(8)

'''
assert not (constantv and randomv) #need to choose between the two

if not constantv and not randomv:
    constantv = True #if you haven't decide we use constant velocity

if constantv:
    print("WARNING: will use fixed constant velocity to infer parameters!")
    time.sleep(15)

if randomv:
    print("WARNING: will use random velocity to infer parameters")
    npseed = seed + parapid
    np.random.seed(npseed)
    print(f"Seeded random num generator with {npseed}")
    time.sleep(15)
'''

assert constantv and not randomv #for evaluator2 you can only use constantv

#choose between (lambda,mu,v), (sqrtDT, sqrtTp), (D,p,T)
assert outputmode == 3 or outputmode == 4 or outputmode == 5

if outputmode == 3:
    convertfun = convert_3
    convertfun_reverse = convert_3_reverse
elif outputmode == 4:
    convertfun = convert_4
    convertfun_reverse = convert_4_reverse
elif outputmode == 5:
    convertfun = convert_5
    convertfun_reverse = convert_5_reverse
else:
    raise Exception("outputmode for growth parameters invalid")

#TODO: currently only works with val set, code needs to be modified to work with separate test set!

start_index = start - 80000
end_index = end - 80000
assert start_index >= 0 and start_index <= 20000
assert end_index >= 0 and end_index <= 20000
assert start_index < end_index

with np.load("results_2.npz") as resultsxyz: #constaint location information
    ys_xyz = resultsxyz['ys'][start_index: end_index, :]
    yspredicted_xyz = resultsxyz['yspredicted'][start_index: end_index, :]

with np.load("results_" + str(outputmode) +".npz") as results_growth:
    ys_growth = results_growth['ys'][start_index : end_index, :]
    yspredicted_growth = results_growth['yspredicted'][start_index : end_index, :]

datapath = "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/"
thrpath = "/mnt/Drive2/ivan_kevin/thresholds/files"
necroticpath = "/mnt/Drive2/ivan_kevin/thresholds/necroticthrs"

all_paths = sorted(glob("{}/*/".format(datapath)))[start : end]
thr_paths = sorted(glob("{}/*".format(thrpath)))[start : end]
necrotic_paths = sorted(glob("{}/*".format(necroticpath)))[start : end]

assert len(all_paths) == len(thr_paths) == len(necrotic_paths)
print(len(all_paths))

print("l2 and l4 should always be equal (except for rounding error)!! IF NOT, THEN YOU ARE PREDICTING ON THE WRONG TUMOR!")

for i in range(0, len(all_paths)):
    print(i)
    path = all_paths[i]
    y_xyz = ys_xyz[i]
    ypredicted_xyz = yspredicted_xyz[i]
    y_growth = ys_growth[i]
    ypredicted_growth = yspredicted_growth[i]

    print(f"path: {path}")
    print(f"y_xyz: {y_xyz}")
    print(f"ypredicted_xyz: {ypredicted_xyz}")
    print(f"y_growth: {y_growth}")
    print(f"ypredicted_growth: {ypredicted_growth}")

    with open(path + "parameter_tag2.pkl", "rb") as par:
        params = pickle.load(par)

    gt_inrange_inpkl_all = (params['Dw'], params['rho'], params['Tend'], params['icx'], params['icy'], params['icz'])
    gt_inrange_inpkl = convertfun_reverse(params['Dw'], params['rho'], params['Tend'], params['icx'], params['icy'], params['icz'])

    if outputmode == 5:
        gt_inrange_innpz = convertfun(y_growth[0], y_growth[1], y_growth[2], y_xyz[0], y_xyz[1], y_xyz[2])
        gt_inrange_innpz_growthloc = convertfun_reverse(gt_inrange_innpz[0], gt_inrange_innpz[1], gt_inrange_innpz[2], gt_inrange_innpz[3],
                                                        gt_inrange_innpz[4], gt_inrange_innpz[5])
    else:
        gt_inrange_innpz = convertfun(y_growth[0], y_growth[1], y_xyz[0], y_xyz[1], y_xyz[2])
        gt_inrange_innpz_growthloc = convertfun_reverse(gt_inrange_innpz[0], gt_inrange_innpz[1], gt_inrange_innpz[2],
                                                        gt_inrange_innpz[3], gt_inrange_innpz[4], gt_inrange_innpz[5])

    #the code below is the most important: it converts predicted parameters by net to (D,p,T)! Code above is just for printing it below
    if not isdebug:
        if outputmode == 5:
            predicted_inrange = convertfun(ypredicted_growth[0], ypredicted_growth[1], ypredicted_growth[2], ypredicted_xyz[0],
                                        ypredicted_xyz[1], ypredicted_xyz[2])
        else:
            predicted_inrange = convertfun(ypredicted_growth[0], ypredicted_growth[1], ypredicted_xyz[0],
                                           ypredicted_xyz[1], ypredicted_xyz[2])
    else:
        print("WARNING: you are in debug mode! RESULTS WILL HAVE PERFECT SCORE!")
        time.sleep(15)
        predicted_inrange = gt_inrange_innpz

    #these calculations are kind of redundant and the above lines are more complicated than they could be,
    #however it gives a good panoramic view of what is being extracted
    #from results.npz, what is in the original pkl file and what was predicted by the network. no time to
    #make it more beautiful unfortunately.
    print(f"l1: GROUND TRUTH (all) in pkl: {gt_inrange_inpkl_all}")
    print(f"l2: GROUND TRUTH (growth + location) in pkl: {gt_inrange_inpkl}")
    print(f"l3: GROUND TRUTH (all) in results.npz (with fixed v when not in debug mode): {gt_inrange_innpz}")
    print(f"l4: GROUND TRUTH (growth + location) in results.npz: {gt_inrange_innpz_growthloc}")
    print(f"PREDICTED (growth + location): {convertfun_reverse(predicted_inrange[0], predicted_inrange[1], predicted_inrange[2], predicted_inrange[3], predicted_inrange[4], predicted_inrange[5])}")
    print(f"PREDICTED (all): {predicted_inrange} with constantv = {constantv}")

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
        necr_path = necrotic_paths[i]
        print(f"thr_path: {thr_path}")
        print(f"necr_path: {necr_path}")
        with np.load(thr_path) as thresholdsfile:
            t1gd_thr = thresholdsfile['t1gd'][0]
            flair_thr = thresholdsfile['flair'][0]
            assert t1gd_thr >= 0.5 and t1gd_thr <= 0.85
            assert flair_thr >= 0.05 and flair_thr <= 0.5

        with np.load(necr_path) as necroticfile:
            necrotic_thr = necroticfile['necrotic'][0]
            assert necrotic_thr >= 0.95 and necrotic_thr <= 1.0

        print(f"t1gd_thr: {t1gd_thr}, flair_thr: {flair_thr}, necrotic_thr: {necrotic_thr}")
        t1gd_volume = (gt_tumor >= t1gd_thr).astype(float)
        flair_volume = (gt_tumor >= flair_thr).astype(float)
        thresholded_volume = 0.666 * t1gd_volume + 0.333 * flair_volume

        pet_volume = ((gt_tumor >= t1gd_thr) * gt_tumor)
        pet_volume = (pet_volume <= necrotic_thr) * pet_volume
        pet_volume_max = pet_volume.max()
        assert pet_volume_max >= 0.0
        if pet_volume_max == 0.0:
            print(f"LIGHT WARNING: empty pet volume for {path}")
            # no division by max, volume is left empty
        else:
            pet_volume = pet_volume / pet_volume.max()
        #pet_volume = ((gt_tumor >= t1gd_thr) * gt_tumor) / 0.5

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
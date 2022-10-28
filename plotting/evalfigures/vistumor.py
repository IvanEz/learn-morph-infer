import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import argparse
from glob import glob
from mayavi import mlab

parser = argparse.ArgumentParser()
parser.add_argument('--identifier', default=0, type=int)
parser.add_argument('--zslice', default=0, type=int)
args = parser.parse_args()
print(args)

identifier = args.identifier
zslice = args.zslice
simulated = glob("/mnt/Drive2/ivan_kevin/29-1-testset-evaluation/inverse_tumor_surrogate/fullpipeline_testsetcode/npzresults/sim/{}-*-Data_0001.npz".format(identifier))

print(simulated)
assert len(simulated) == 1

simulated = simulated[0]

toevaluatepaths = sorted(glob("{}/*/".format("/mnt/Drive2/ivan_kevin/samples_extended_thr2_testset/")))[0 : 12000]

groundtruth = toevaluatepaths[identifier]

thresholds = np.load("/mnt/Drive2/ivan_kevin/29-1-testset-evaluation/inverse_tumor_surrogate/fullpipeline_testsetcode/scanthresholds.npz")

t1gd = thresholds['t1gd'][0:12000][identifier]
flair = thresholds['flair'][0:12000][identifier]
necrotic = thresholds['necrotic'][0:12000][identifier]

print(f"{groundtruth}, {t1gd}, {flair}, {necrotic}")

gt_tumor = np.load(groundtruth + "Data_0001_thr2.npz")['data']
sim_tumor = np.load(simulated)['data'][:,:,:,0]
sim_tumor_brain = np.load(simulated)['data'][:,:,:,1]


t1gd_scan = (gt_tumor >= t1gd).astype(float)
flair_scan = (gt_tumor >= flair).astype(float)
mri_scan = 0.666 * t1gd_scan + 0.333 * flair_scan

pet_scan = (gt_tumor >= t1gd) * gt_tumor
pet_scan = (pet_scan <= necrotic) * pet_scan

if pet_scan.max() != 0.0:
    pet_scan = pet_scan / pet_scan.max()

f=mlab.figure(); mlab.volume_slice(mri_scan,figure=f,plane_orientation='z_axes',slice_index=zslice, vmin=0.0, vmax=0.999)
f=mlab.figure(); mlab.volume_slice(pet_scan,figure=f,plane_orientation='z_axes',slice_index=zslice)
f=mlab.figure(); mlab.volume_slice(gt_tumor,figure=f,plane_orientation='z_axes',slice_index=zslice,vmin=0.0,vmax=1.0)
f=mlab.figure(); mlab.volume_slice(sim_tumor,figure=f,plane_orientation='z_axes',slice_index=zslice,vmin=0.0,vmax=1.0)
f=mlab.figure(); mlab.volume_slice(sim_tumor_brain,figure=f,plane_orientation='z_axes',slice_index=zslice,vmin=0.0,vmax=1.2)
mlab.show()

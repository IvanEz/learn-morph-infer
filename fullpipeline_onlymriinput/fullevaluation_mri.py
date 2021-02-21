import numpy as np
import argparse
from glob import glob
import pickle5 as pickle
import subprocess
import time
import torch
torch.set_num_threads(2) #uses max. 2 cpus for inference! no gpu inference!
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from datetime import datetime
##################################
#
# DON'T FORGET TO CALL "source ./setup_ibbm_giga.sh" so libraries work!
#
##################################
normalization_range = [-1.0, 1.0]
thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 0.9, 0.95]


parser = argparse.ArgumentParser()
#args
#parser.add_argument('--cpu', action='store_true') #if you use --cpu, gpu flag will be ignored and cpu will be used
#parser.add_argument('--gpu', default=4, type=int)
parser.add_argument('--datadir', default="/mnt/Drive2/ivan_kevin/testsetdata", type=str) #directory containing data you want to evaluate
parser.add_argument('--is_ugt', action='store_true') #if is_ugt, then creates synthetic observations D_atlas
parser.add_argument('--thresholdsdir', default="/mnt/Drive2/ivan_kevin/testsetdata", type=str) #directory containing thresholds when is_ugt and synthetic observations D_atlas need to be created
parser.add_argument('--start', default=0, type=int) #inclusive
parser.add_argument('--stop', default=1, type=int) #exclusive
parser.add_argument('--printeverything', action='store_true')
parser.add_argument('--parapid', default=0, type=int)
parser.add_argument('--name', default="noname", type=str)
args = parser.parse_args()
print(args)
#cpu = args.cpu
#gpu = args.gpu
datadir = args.datadir
is_ugt = args.is_ugt
thresholdsdir = args.thresholdsdir
start = args.start
stop = args.stop
printeverything = args.printeverything
parapid = args.parapid
name = args.name

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


#if cpu:
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
#else:
#    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def conv3x3_biased(in_planes, out_planes, stride=1, padding=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=True)

class BasicBlockInv_Pool_constant_noBN_n4_inplace(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_Pool_constant_noBN_n4_inplace, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        #self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3_biased(inplanes, inplanes)
        self.relu1 = torch.nn.ReLU(inplace=True)

        #self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.conv2 = conv3x3_biased(inplanes, inplanes)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.conv3 = conv3x3_biased(inplanes, inplanes)
        self.relu3 = torch.nn.ReLU(inplace=True)

        self.conv4 = conv3x3_biased(inplanes, inplanes)
        self.relu4 = torch.nn.ReLU(inplace=True)


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)

        #out = self.bn1(x)
        out = self.conv1(x)
        out = self.relu1(out)

        #out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = out + x
        #out += x

        return out

class NetConstant_noBN_l4_inplacefull(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels, includesft=False):
        super(NetConstant_noBN_l4_inplacefull, self).__init__()

        if not includesft:
            self.inplanes = 1  # initial number of channels
        else:
            raise Exception("no ft")

        self.conv1_i = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1_i = torch.nn.ReLU(inplace=True)
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        #self.bn_final = torch.nn.BatchNorm3d(channels)
        #self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                #torch.nn.init.constant_(m.weight, 1)
                #torch.nn.init.constant_(m.bias, 0)
                raise Exception("no batchnorm")
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, blocks, downsample=True):
        layers = []
        layers.append(block(self.inplanes, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_i(x)
        x = self.relu1_i(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)

        #x = self.bn_final(x)
        #x = self.relu_final(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

def NetConstant_noBN_64_n4_l4_inplacefull(numoutputs):
    return NetConstant_noBN_l4_inplacefull(BasicBlockInv_Pool_constant_noBN_n4_inplace, [1,1,1,1], numoutputs, 64)

xyz_model = NetConstant_noBN_64_n4_l4_inplacefull(3)
growth_model = NetConstant_noBN_64_n4_l4_inplacefull(2)

xyz_checkpoint = torch.load("/mnt/Drive2/ivan_kevin/log/torchimpl/0502-13-56-41-v7_1_mri-batch32-wd05-xyz/epoch66.pt", map_location=torch.device('cpu'))
xyz_model.load_state_dict(xyz_checkpoint['model_state_dict'])

growth_checkpoint = torch.load("/mnt/Drive2/ivan_kevin/log/torchimpl/1302-19-01-21-v7_1_mri-nonorm-batch32-wd1-mu1-mu2/epoch47.pt", map_location=torch.device('cpu'))
growth_model.load_state_dict(growth_checkpoint['model_state_dict'])

xyz_model = xyz_model.eval()
growth_model = growth_model.eval()

#different than before, if you choose is_ugt, the thresholds are going to be in ONE single file, before
#every datapoint had its own file with all thresholds for training
#3 files: t1gd, flair, necrotic that need to at least as many thresholds as you have datapoints

data_paths = sorted(glob("{}/*/".format(datadir)))[start : stop] #data that will be processed by this worker
if printeverything:
    print(f"data_paths: {data_paths}")
    from mayavi import mlab
#to measure time: with torch.autograd.profiler.profile() as prof:
#time in seconds spent: prof.self_cpu_time_total / (1000*1000)

if is_ugt:
    thresholds_scans = np.load(thresholdsdir + "/scanthresholds.npz")
    t1gd_thresholds = thresholds_scans['t1gd'][start : stop]
    flair_thresholds = thresholds_scans['flair'][start : stop]
    #necrotic_thresholds = thresholds_scans['necrotic'][start : stop]

    if printeverything:
        print(f"t1gd_thresholds: {t1gd_thresholds}, flair_thresholds: {flair_thresholds}")

    assert len(t1gd_thresholds) == stop - start
    assert len(flair_thresholds) == stop - start
    #assert len(necrotic_thresholds) == stop - start


resultstosave = []
for (i, path) in enumerate(data_paths):
    if is_ugt:
        print(path + "Data_0001_thr2.npz")
        with np.load(path + "Data_0001_thr2.npz") as data:
            volume = data['data']
            t1gd = t1gd_thresholds[i]
            flair = flair_thresholds[i]
            #necrotic = necrotic_thresholds[i]
            print(f"t1gd: {t1gd}, flair: {flair}")
            t1gd_scan = (volume >= t1gd).astype(float)
            flair_scan = (volume >= flair).astype(float)
            mri_scan = 0.666 * t1gd_scan + 0.333 * flair_scan

            #pet_scan = (volume >= t1gd) * volume
            #pet_scan = (pet_scan <= necrotic) * pet_scan

            #if pet_scan.max() != 0.0:
            #    pet_scan = pet_scan / pet_scan.max()
    else:
        with np.load(path + "tumor_scans.npz") as data:
            mri_scan = data['mri']
            #pet_scan = data['pet']

    #load pickle

    if printeverything:
        if is_ugt:
            f = mlab.figure(); mlab.volume_slice(volume, figure=f)
            mlab.title("gt", figure=f)

        f=mlab.figure(); mlab.volume_slice(mri_scan,figure=f)
        mlab.title("mri", figure=f)
        #f=mlab.figure(); mlab.volume_slice(pet_scan,figure=f)
        #mlab.title("pet", figure=f)

    mri_scan = np.expand_dims(mri_scan, -1)
    #pet_scan = np.expand_dims(pet_scan, -1)

    #nn_input = np.concatenate((mri_scan, pet_scan), -1)
    nn_input = mri_scan
    nn_input = nn_input.transpose((3, 0, 1, 2))

    if printeverything:
        print(nn_input.shape)

    nn_input = torch.from_numpy(nn_input.astype(np.float32))
    nn_input = nn_input.unsqueeze(0) #add artificial batch dimension=1

    if printeverything:
        print(nn_input.shape)

    with torch.set_grad_enabled(False):
        with torch.autograd.profiler.profile() as prof:
            predicted_xyz = xyz_model(nn_input)
            predicted_growth = growth_model(nn_input)

    print(f"time spent: {prof.self_cpu_time_total / (1000*1000)} s")

    predicted_xyz = predicted_xyz.numpy()
    predicted_growth = predicted_growth.numpy()

    print(f"xyz: {predicted_xyz}, growth: {predicted_growth}")

    predicted_xyz = predicted_xyz[0]
    predicted_growth = predicted_growth[0]

    icx_predicted = np.interp(predicted_xyz[0], normalization_range, [0.15, 0.7])
    icy_predicted = np.interp(predicted_xyz[1], normalization_range, [0.2, 0.8])
    icz_predicted = np.interp(predicted_xyz[2], normalization_range, [0.15, 0.7])

    mu1_predicted = np.interp(predicted_growth[0], normalization_range, [0.1, np.sqrt(22.5)]) #cm
    mu2_predicted = np.interp(predicted_growth[1], normalization_range, np.sqrt([0.1, 300.0])) #constant

    print(f"mu1_predicted: {mu1_predicted}, mu2_predicted: {mu2_predicted}, icx_predicted: {icx_predicted}, icy_predicted: {icy_predicted}, icz_predicted: {icz_predicted}")

    v = np.interp(np.mean(normalization_range), normalization_range, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)])

    D_predicted = (mu1_predicted * v) / (2.0 * mu2_predicted)  # cm^2 / d
    p_predicted = (mu2_predicted * v) / (2.0 * mu1_predicted)  # 1 / d
    T_predicted = (2.0 * mu1_predicted * mu2_predicted) / v  # days

    command = "./brain -model RD -PatFileName /mnt/Drive2/ivan_kevin/differentv/anatomy_dat/ -Dw " + str(
        D_predicted) + " -rho " + str(p_predicted) + " -Tend " + str(T_predicted) + " -dumpfreq " + str(0.9999 * T_predicted) + " -icx " + str(
        icx_predicted) + " -icy " + str(icy_predicted) + " -icz " + str(icz_predicted) + " -vtk 1 -N 16 -adaptive 0"

    print(f"command: {command}")

    currenttime = datetime.now()
    currenttime = currenttime.strftime("%d%m-%H-%M-%S-")

    simulation = subprocess.check_call([command], shell=True, cwd="./vtus" + str(parapid) + "/sim/")  # e.g. ./vtus0/sim/
    tumorname = str(start + i) + "-" + currenttime
    vtu2npz = subprocess.check_call(["python3 -u vtutonpz3.py --parapid " + str(parapid) + " --name " + tumorname], shell=True)

    with np.load("./npzresults/sim/" + tumorname + "Data_0001.npz") as inferredtumor:
        u_sim = inferredtumor['data'][:,:,:,0]

    if printeverything:
        f=mlab.figure(); mlab.volume_slice(u_sim); mlab.title("u_sim", figure=f)

    if is_ugt:
        with open(path + "parameter_tag2.pkl", "rb") as par:
            params = pickle.load(par)
            Dw_gt = params['Dw']  # cm^2 / d
            rho_gt = params['rho']  # 1 / d
            Tend_gt = params['Tend']  # d
            icx_gt = params['icx']
            icy_gt = params['icy']
            icz_gt = params['icz']

        mu1_gt = np.sqrt(Dw_gt * Tend_gt) #cm
        mu2_gt = np.sqrt(rho_gt * Tend_gt) #constant

        print(f"GT time-independent -> mu_1_gt = {mu1_gt}, mu_2_gt = {mu2_gt}, icx_gt = {icx_gt}, icy_gt = {icy_gt}, icz_gt = {icz_gt}")
        print(f"Other GT parameters -> Dw_gt =  {Dw_gt}, rho_gt = {rho_gt}, Tend_gt = {Tend_gt}")

        mu1_absolute_error = np.abs(mu1_predicted - mu1_gt)
        mu2_absolute_error = np.abs(mu2_predicted - mu2_gt)
        icx_absolute_error = np.abs(icx_predicted - icx_gt)
        icy_absolute_error = np.abs(icy_predicted - icy_gt)
        icz_absolute_error = np.abs(icz_predicted - icz_gt)

        mu1_relative_error = np.abs(1 - (mu1_predicted / mu1_gt))
        mu2_relative_error = np.abs(1 - (mu2_predicted / mu2_gt))

        print(f"mu1_absolute_error = {mu1_absolute_error}, mu2_absolute_error = {mu2_absolute_error}, "
              f"mu1_relative_error = {mu1_relative_error}, mu2_relative_error = {mu2_relative_error}, "
              f"icx_absolute_error = {icx_absolute_error}, icy_absolute_error = {icy_absolute_error}, "
              f"icz_absolute_error = {icz_absolute_error}")

        diceresults = dice(volume, u_sim) #volume is u_gt

        print(f"diceresults = {diceresults}")

        resultstosave.append({"t1gd": t1gd, "flair": flair,
                              "mu1_gt": mu1_gt, "mu2_gt": mu2_gt,
                              "mu1_predicted": mu1_predicted, "mu2_predicted": mu2_predicted,
                              "icx_gt": icx_gt, "icy_gt": icy_gt, "icz_gt": icz_gt,
                              "icx_predicted": icx_predicted, "icy_predicted": icy_predicted, "icz_predicted": icz_predicted,
                              "diceresults": diceresults, "identifier": start + i, "path": path
                              })

    if printeverything:
        mlab.show()

currenttime = datetime.now()
currenttime = currenttime.strftime("%d%m-%H-%M-%S-")

#with open("evaluation-" + str(parapid) + "-" + currenttime + "start-" + str(start) + "-stop-" + str(stop) + "-" + name + ".pkl", "wb") as savefile:
#    pickle.dump(resultstosave, savefile)

with open("evaluation" + str(parapid) + ".pkl", "wb") as savefile:
    pickle.dump(resultstosave, savefile)

#np.savez_compressed("evaluation-" + str(parapid) + "-" + currenttime + "start-" + str(start) + "-stop-" + str(stop) + "-" + name, results=np.array(resultstosave))

print("Terminated successfully")
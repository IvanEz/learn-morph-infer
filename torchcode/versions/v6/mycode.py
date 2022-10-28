import torch
import numpy as np
import pickle5 as pickle
import matplotlib.pyplot as plt
import sys
import os
from glob import glob
#e.g. Dataset("/home/kevin/Desktop/thresholding/", 0, 9)
normalization_range = [-1.0, 1.0]

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datapath, beginning, ending, thrpath, num_thresholds=100):
        'Initialization'
        #beginning inclusive, ending exclusive
        self.datapath = datapath
        self.beginning = beginning
        self.ending = ending
        self.thrpath = thrpath
        self.all_paths = sorted(glob("{}/*/".format(self.datapath)))[self.beginning : self.ending]
        self.threshold_paths = sorted(glob("{}/*".format(self.thrpath)))[self.beginning : self.ending]
        self.datasetsize = len(self.all_paths)
        self.epoch = 0
        self.num_thresholds = num_thresholds
        #self.epoch determines which thresholds are used, in train set modified so we have different thresholds,
        #in validation set we DO NOT INCREMENT this so the same thresholds are always used for every epoch
        #If you only want one threshold even in training: set num_thresholds=1! Then one threshold pair per datapoint
        #like valset!
        assert len(self.all_paths) == len(self.threshold_paths)

        #VERY IMPORTANT: we use the sorted() function to sort paths! This just makes the ordering look weird but started
        #with this function so have to stick to it.

  def __len__(self):
        'Denotes the total number of samples'
        #return (self.ending - self.beginning)
        return self.datasetsize

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        file_path = self.all_paths[index]
        thr_path = self.threshold_paths[index]

        with np.load(thr_path) as thresholdsfile:
            t1gd_thr = thresholdsfile['t1gd'][self.epoch % self.num_thresholds]
            flair_thr = thresholdsfile['flair'][self.epoch % self.num_thresholds]

        #print("got pos " + str(index) + " which corresponds to " + str(file_path))
        with np.load(file_path + "Data_0001_thr2.npz") as data:
            #thrvolume = data['thr2_data']
            volume = data['data']
            volume_resized = np.delete(np.delete(np.delete(volume, 128, 0), 128, 1), 128, 2) #from 129x129x129 to 128x128x128
            #TODO: check if deletion removed nonzero entries (especially last slice: thrvolume[...][...][128])

            #t1gd_thr = round(0.35 * self.rngs[index].rand() + 0.5, 5)
            #flair_thr = round(0.45 * self.rngs[index].rand() + 0.05, 5)

            t1gd_volume = (volume_resized >= t1gd_thr).astype(float)
            flair_volume = (volume_resized >= flair_thr).astype(float)

            input_volume = 0.666 * t1gd_volume + 0.333 * flair_volume

            thrvolume_resized = np.expand_dims(input_volume, -1) #now it is 128x128x128x1
            #print(thrvolume_resized.shape)

        with open(file_path + "parameter_tag2.pkl", "rb") as par:
            #TODO: interpolate with manual formulas (e.g. uth: 10x - 7)
            #TODO: rounding to 6 digits?
            paramsarray = np.zeros(8)
            params = pickle.load(par)
            paramsarray[0] = np.interp(t1gd_thr, [0.5, 0.85], normalization_range)
            paramsarray[1] = np.interp(flair_thr, [0.05, 0.5], normalization_range)
            paramsarray[2] = np.interp(params['Dw'], [0.0002, 0.015], normalization_range)
            paramsarray[3] = np.interp(params['rho'], [0.002, 0.2], normalization_range)
            paramsarray[4] = np.interp(params['Tend'], [50, 1500], normalization_range)
            paramsarray[5] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
            paramsarray[6] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
            paramsarray[7] = np.interp(params['icz'], [0.15, 0.7], normalization_range)

        thrvolume_resized = thrvolume_resized.transpose((3,0,1,2))
        return torch.from_numpy(thrvolume_resized.astype(np.float32)), torch.from_numpy(paramsarray.astype(np.float32))

##########################################################################################################

class Dataset2(Dataset):
    # We remove tanh from last layer when predicting infiltration length + Tp + velocity, because mean of Dw and p
    # after bringing into range [-1, 1] was at 0. Since we now predict products of these factors, we can observe
    # our data and see that when we normalize into [-1, 1] range, the mean (of our TRAINING DATA) is not at 0 anymore!
    def __init__(self, datapath, beginning, ending, thrpath, num_thresholds=100, includesft=False, outputmode=0):
        Dataset.__init__(self, datapath, beginning, ending, thrpath, num_thresholds=num_thresholds)
        self.includesft = includesft
        self.outputmode = outputmode

    def __len__(self):
        return self.datasetsize

    def __getitem__(self, index):
        file_path = self.all_paths[index]
        thr_path = self.threshold_paths[index]

        with np.load(thr_path) as thresholdsfile:
            t1gd_thr = thresholdsfile['t1gd'][self.epoch % self.num_thresholds]
            flair_thr = thresholdsfile['flair'][self.epoch % self.num_thresholds]

        # print("got pos " + str(index) + " which corresponds to " + str(file_path))
        with np.load(file_path + "Data_0001_thr2.npz") as data:
            # thrvolume = data['thr2_data']
            volume = data['data']
            volume_resized = volume
            #volume_resized = np.delete(np.delete(np.delete(volume, 128, 0), 128, 1), 128, 2)  # from 129x129x129 to 128x128x128
            # TODO: check if deletion removed nonzero entries (especially last slice: thrvolume[...][...][128])

            # t1gd_thr = round(0.35 * self.rngs[index].rand() + 0.5, 5)
            # flair_thr = round(0.45 * self.rngs[index].rand() + 0.05, 5)

            t1gd_volume = (volume_resized >= t1gd_thr).astype(float)
            flair_volume = (volume_resized >= flair_thr).astype(float)

            thr_volume = 0.666 * t1gd_volume + 0.333 * flair_volume

            thrvolume_resized = np.expand_dims(thr_volume, -1)  # now it is 129x129x129x1
            #print(thrvolume_resized.shape)

            b = 0.5

            pet_volume = ((volume_resized >= t1gd_thr) * volume_resized) / b
            #print(pet_volume.shape)
            pet_volume_reshaped = np.expand_dims(pet_volume, -1) #now 129x129x129x1
            #print(pet_volume_reshaped.shape)

            nn_input = np.concatenate((thrvolume_resized, pet_volume_reshaped), -1)
            #print(nn_input.shape)

            if self.includesft:
                ft = np.abs(np.fft.fftshift(np.fft.fftn(thr_volume + pet_volume, norm='ortho')))
                ft_reshaped = np.expand_dims((ft / np.max(ft)), -1)
                #nn_input = np.concatenate((nn_input, ft_reshaped), -1)
                nn_input = ft_reshaped  # OVERWRITES NN_INPUT, IS NOW ONLY FOURIER TRANSFORM, NOT SPATIAL TUMOR!
                if index == 0:
                    print("Shape is " + str(nn_input.shape) + ", should be (129,129,129,1)")

        with open(file_path + "parameter_tag2.pkl", "rb") as par:
            # TODO: interpolate with manual formulas (e.g. uth: 10x - 7)
            # TODO: rounding to 6 digits?
            params = pickle.load(par)

            Dw = params['Dw'] #cm^2 / d
            rho = params['rho'] #1 / d
            Tend = params['Tend'] #d

            lambdaw = np.sqrt(Dw / rho) #cm
            mu = Tend * rho #constant
            velocity = 2 * np.sqrt(Dw * rho) #cm / d

            if self.outputmode == 0:
                paramsarray = np.zeros(8)
                paramsarray[0] = np.interp(t1gd_thr, [0.5, 0.85], normalization_range)
                paramsarray[1] = np.interp(flair_thr, [0.05, 0.5], normalization_range)
                paramsarray[2] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[3] = np.interp(mu, [0.1, 300.0], normalization_range)
                paramsarray[4] = np.interp(velocity, [2*np.sqrt(4e-7), 2*np.sqrt(0.003)], normalization_range)
                paramsarray[5] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
                paramsarray[6] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
                paramsarray[7] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            elif self.outputmode == 1:
                paramsarray = np.zeros(3)
                paramsarray[0] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[1] = np.interp(mu, [0.1, 300.0], normalization_range)
                paramsarray[2] = np.interp(velocity, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)], normalization_range)
            elif self.outputmode == 2:
                paramsarray = np.zeros(3)
                paramsarray[0] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
                paramsarray[1] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
                paramsarray[2] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            elif self.outputmode == 3:
                paramsarray = np.zeros(2)
                paramsarray[0] = np.interp(lambdaw, [np.sqrt(0.001), np.sqrt(7.5)], normalization_range)
                paramsarray[1] = np.interp(mu, [0.1, 300.0], normalization_range)
            else:
                raise Exception("invalid output mode")

            '''
            paramsarray[0] = np.interp(t1gd_thr, [0.5, 0.85], normalization_range)
            paramsarray[1] = np.interp(flair_thr, [0.05, 0.5], normalization_range)
            paramsarray[2] = np.interp(params['Dw'], [0.0002, 0.015], normalization_range)
            paramsarray[3] = np.interp(params['rho'], [0.002, 0.2], normalization_range)
            paramsarray[4] = np.interp(params['Tend'], [50, 1500], normalization_range)
            paramsarray[5] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
            paramsarray[6] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
            paramsarray[7] = np.interp(params['icz'], [0.15, 0.7], normalization_range)
            '''

        nninput_resized = nn_input.transpose((3, 0, 1, 2))
        return torch.from_numpy(nninput_resized.astype(np.float32)), torch.from_numpy(paramsarray.astype(np.float32))

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv3x3_biased(in_planes, out_planes, stride=1, padding=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=True)

def conv1x1(in_planes, out_planes, stride=1, padding=0):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)

class ConvNet(torch.nn.Module):

    def __init__(self, numoutputs, dropoutrate=0.2):
        super(ConvNet, self).__init__()

        self.seq = torch.nn.Sequential(
                    conv3x3(2,2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(2),
                    conv3x3(2,2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(2),
                    #--------still 128---------
                    conv3x3(2,4, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(4),
                    conv3x3(4,4),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(4),
                    #--------still 64-------
                    conv3x3(4,8, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(8),
                    conv3x3(8,8),   
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(8),
                    #-------still 32--------
                    conv3x3(8,16, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(16),
                    conv3x3(16,16),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(16),
                    #-------still 16--------   
                    conv3x3(16,32, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(32),
                    conv3x3(32,32),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(32),
                    #-------still 8----------                           
                    torch.nn.Flatten(),
                    torch.nn.Dropout(p=dropoutrate),
                    torch.nn.Linear(8*8*8*32, numoutputs)
                    )

    def forward(self, x):
        return self.seq(x)

class ConvNet2(torch.nn.Module):

    def __init__(self, numoutputs, dropoutrate=0.2):
        super(ConvNet2, self).__init__()

        self.seq = torch.nn.Sequential(
                    conv3x3(2,2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(2),
                    conv3x3(2,2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(2),
                    #--------still 128---------
                    conv3x3(2,4, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(4),
                    conv3x3(4,4),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(4),
                    #--------still 64-------
                    conv3x3(4,8, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(8),
                    conv3x3(8,8),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(8),
                    #-------still 32--------
                    conv3x3(8,16, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(16),
                    conv3x3(16,16),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(16),
                    #-------still 16--------
                    conv3x3(16,32, stride=2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(32),
                    conv3x3(32,32),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm3d(32),
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                    #-------still 8----------
                    torch.nn.Flatten(),
                    torch.nn.Dropout(p=dropoutrate),
                    torch.nn.Linear(32, numoutputs),
                    torch.nn.Tanh()
                    )

    def forward(self, x):
        return self.seq(x)

#adapted from https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html and https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class BasicBlockInv(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(BasicBlockInv, self).__init__()

        norm_layer = torch.nn.BatchNorm3d
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = torch.nn.ReLU()
        self.stride = stride
        self.downsample = downsample
        if self.downsample:
            self.conv3 = conv1x1(inplanes, planes, stride)
            self.bn3 = torch.nn.BatchNorm3d(planes)

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        #print(f"Layer: {out.mean()}, {out.std()}")
        out = self.bn1(out)
        #print(f"Layer: {out.mean()}, {out.std()}")
        #print(f"bn1_mean: {self.bn1.running_mean.mean()}, bn1_var: {self.bn1.running_var.mean()}, bn1_var_min: {self.bn1.running_var.min()}, bn1_var_max: {self.bn1.running_var.max()}")
        out = self.relu(out)
        #print(f"Layer: {out.mean()}, {out.std()}")

        out = self.conv2(out)
        #print(f"Layer: {out.mean()}, {out.std()}")
        out = self.bn2(out)
        #print(f"Layer: {out.mean()}, {out.std()}")

        if self.downsample is True:
            identity = self.conv3(identity)
            #print(f"Layer downsample: {identity.mean()}, {identity.std()}")
            identity = self.bn3(identity)
            #print(f"Layer downsample relu: {identity.mean()}, {identity.std()}")

        out += identity
        #print(f"Layer after addition: {out.mean()}, {out.std()}")
        out = self.relu2(out)
        #print(f"Layer after relu: {out.mean()}, {out.std()}")

        return out


class BasicBlockInv_PreAct_Pool(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_PreAct_Pool, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = conv3x3(inplanes, inplanes)

        self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = conv3x3(inplanes, inplanes)


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample:
            out = torch.cat((x, out), 1) #double the amount of channels through concatenation
        else:
            out = out + x

        return out

class BasicBlockInv_PreAct_Pool_constant(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_PreAct_Pool_constant, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = conv3x3(inplanes, inplanes)

        self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = conv3x3(inplanes, inplanes)


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = out + x

        return out

class BasicBlockInv_Pool_constant_noBN(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_Pool_constant_noBN, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        #self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3_biased(inplanes, inplanes)
        self.relu1 = torch.nn.ReLU()

        #self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.conv2 = conv3x3_biased(inplanes, inplanes)
        self.relu2 = torch.nn.ReLU()


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)

        #out = self.bn1(x)
        out = self.conv1(x)
        out = self.relu1(out)

        #out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = out + x

        return out

class BasicBlockInv_Pool_constant_withBN(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_Pool_constant_withBN, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = conv3x3(inplanes, inplanes)
        self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.relu2 = torch.nn.ReLU()


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out + x

        return out

class BasicBlockInv_Pool_constant_noBN_n4(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_Pool_constant_noBN_n4, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        #self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3_biased(inplanes, inplanes)
        self.relu1 = torch.nn.ReLU()

        #self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.conv2 = conv3x3_biased(inplanes, inplanes)
        self.relu2 = torch.nn.ReLU()

        self.conv3 = conv3x3_biased(inplanes, inplanes)
        self.relu3 = torch.nn.ReLU()

        self.conv4 = conv3x3_biased(inplanes, inplanes)
        self.relu4 = torch.nn.ReLU()


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

class BasicBlockInv_Pool_constant_n4_inorm(torch.nn.Module):
    def __init__(self, inplanes, downsample=False, normalize=False):
        super(BasicBlockInv_Pool_constant_n4_inorm, self).__init__()
        self.normalize = normalize
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

        if self.normalize:
            self.norm0 = torch.nn.InstanceNorm3d(inplanes)
            self.norm1 = torch.nn.InstanceNorm3d(inplanes)
            self.norm2 = torch.nn.InstanceNorm3d(inplanes)
            self.norm3 = torch.nn.InstanceNorm3d(inplanes)
            self.norm4 = torch.nn.InstanceNorm3d(inplanes)


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)
        if self.normalize:
            x = self.norm0(x)

        out = self.conv1(x)
        if self.normalize:
            out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.normalize:
            out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.normalize:
            out = self.norm3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        if self.normalize:
            out = self.norm4(out)
        out = self.relu4(out)

        out = out + x
        #out += x

        return out


class ResNetInv(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv, self).__init__()

        self.inplanes = 2 #initial number of channels

        self.layer1 = self._make_layer(block, 1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(8*8*8*32, numoutputs)
        self.tanh = torch.nn.Tanh()

        #TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias, 0)
            #elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)
                

    def _make_layer(self, block, planes, blocks, stride):
        downsample = False
        if stride != 1 or self.inplanes != planes:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x,1)
        x = self.do(x)
        #print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        #print(f"Layer after fc: {x.mean()}, {x.std()}")
        #print("Before tanh: " + str(x))
        x = self.tanh(x)
        #print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

class ResNetInv2(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv2, self).__init__()

        self.inplanes = 2 #initial number of channels

        self.layer1 = self._make_layer(block, 2, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 64, layers[5], stride=2)
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(4*4*4*64, numoutputs)
        self.tanh = torch.nn.Tanh()

        #TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias, 0)
            #elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)
                

    def _make_layer(self, block, planes, blocks, stride):
        downsample = False
        if stride != 1 or self.inplanes != planes:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = torch.flatten(x,1)
        x = self.do(x)
        #print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        #print(f"Layer after fc: {x.mean()}, {x.std()}")
        #print("Before tanh: " + str(x))
        x = self.tanh(x)
        #print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x


class ResNetInv2Pool(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv2Pool, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.layer1 = self._make_layer(block, 2, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 64, layers[5], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(64, numoutputs)
        self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = False
        if stride != 1 or self.inplanes != planes:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

class ResNetInv2PoolNL(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv2PoolNL, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.layer1 = self._make_layer(block, 2, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 64, layers[5], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(64, numoutputs)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(numoutputs,numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = False
        if stride != 1 or self.inplanes != planes:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

class ResNetInv2Pool_5(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv2Pool_5, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.layer1 = self._make_layer(block, 2, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(32, numoutputs)
        self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = False
        if stride != 1 or self.inplanes != planes:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

#based on preactivation resnet (see paper+code), removes skip connection with stride 1, replaced by concatenation (see FishNet, DenseNet)
#we get rid of I-convs
class ResNetInvPreActDirect(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInvPreActDirect, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.layer1 = self._make_layer(block, layers[0], doublechannels=False)
        self.layer2 = self._make_layer(block, layers[1], doublechannels=True)
        self.layer3 = self._make_layer(block, layers[2], doublechannels=True)
        self.layer4 = self._make_layer(block, layers[3], doublechannels=True)
        self.layer5 = self._make_layer(block, layers[4], doublechannels=True)

        self.bn_final = torch.nn.BatchNorm3d(32)
        self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(32, numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, blocks, doublechannels=False):
        downsample = doublechannels

        layers = []
        layers.append(block(self.inplanes, downsample))
        self.inplanes = 2 * self.inplanes if downsample else self.inplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.bn_final(x)
        x = self.relu_final(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

#intended for bigger num of channels at beginning
class ResNetInvPreActDirect_Wider_2(torch.nn.Module):
    def __init__(self, block, constantblock, layers, numoutputs, dropoutrate, channels):
        super(ResNetInvPreActDirect_Wider_2, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=False)
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False, doubling=False)
        self.layer2 = self._make_layer(block, layers[1], downsample=True, doubling=True)
        self.layer3 = self._make_layer(block, layers[2], downsample=True, doubling=True)
        self.layer4 = self._make_layer(constantblock, layers[3], downsample=True)
        self.layer5 = self._make_layer(constantblock, layers[4], downsample=True)

        self.bn_final = torch.nn.BatchNorm3d(channels * 4)
        self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(channels * 4, numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, blocks, downsample=False, doubling=False):
        layers = []
        layers.append(block(self.inplanes, downsample))
        self.inplanes = 2 * self.inplanes if doubling else self.inplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.bn_final(x)
        x = self.relu_final(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

#intended for bigger num of channels at beginning
class PreActNetConstant(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels):
        super(PreActNetConstant, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=False)
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        self.layer5 = self._make_layer(block, layers[4])

        self.bn_final = torch.nn.BatchNorm3d(channels)
        self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
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
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.bn_final(x)
        x = self.relu_final(x)

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

class NetConstant_noBN(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels):
        super(NetConstant_noBN, self).__init__()

        self.inplanes = 2  # initial number of channels

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        self.layer5 = self._make_layer(block, layers[4])

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
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

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

class NetConstant_noBN_l4(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels, includesft=False):
        super(NetConstant_noBN_l4, self).__init__()

        if not includesft:
            self.inplanes = 2  # initial number of channels
        else:
            self.inplanes = 3

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1 = torch.nn.ReLU()
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
        x = self.conv1(x)
        x = self.relu1(x)

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

class NetConstant_noBN_l4_separateexperiment(torch.nn.Module): #ALWAYS INCLUDES FT
    def __init__(self, block, layers, numoutputs, channels):
        super(NetConstant_noBN_l4_separateexperiment, self).__init__()

        self.channels = channels
        ###### SPACE FEATURE EXTRACTION ######
        self.conv1 = torch.nn.Conv3d(2, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1 = torch.nn.ReLU()

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        #self.bn_final = torch.nn.BatchNorm3d(channels)
        #self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        #self.fc = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()
        ###### SPACE FEATURE EXTRACTION ######      

        ###### FREQUENCY FEATURE EXTRACTION ######

        self.conv1_f = torch.nn.Conv3d(1, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1_f = torch.nn.ReLU()

        self.layer1_f = self._make_layer(block, layers[0], downsample=False)
        self.layer2_f = self._make_layer(block, layers[1])
        self.layer3_f = self._make_layer(block, layers[2])
        self.layer4_f = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        #self.bn_final = torch.nn.BatchNorm3d(channels)
        #self.relu_final = torch.nn.ReLU()
        self.avgpool_f = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        #self.fc_f = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()
        ###### FREQUENCY FEATURE EXTRACTION ######     

        self.fc_both = torch.nn.Linear(2*self.channels, numoutputs)

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
        layers.append(block(self.channels, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.channels))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        #fourier = x[:,2,:,:,:][:,None,:,:,:] #extracts fourier transform from input
        #tumor = torch.cat((x[:,0,:,:,:][:,None,:,:,:], x[:,1,:,:,:][:,None,:,:,:]), dim=1)

        x = torch.chunk(x,2,dim=1)
        tumor = x[0]
        fourier = x[1]

        #print(fourier.shape)
        #print(tumor.shape)
        #print("finished separating input x")

        tumor = self.conv1(tumor)
        fourier = self.conv1_f(fourier)

        tumor = self.relu1(tumor)
        fourier = self.relu1_f(fourier)

        tumor = self.layer1(tumor)
        fourier = self.layer1_f(fourier)

        tumor = self.layer2(tumor)
        fourier = self.layer2_f(fourier)

        tumor = self.layer3(tumor)
        fourier = self.layer3_f(fourier)

        tumor = self.layer4(tumor)
        fourier = self.layer4_f(fourier)
        #x = self.layer5(x)

        #x = self.bn_final(x)
        #x = self.relu_final(x)

        tumor = self.avgpool(tumor)
        tumor = torch.flatten(tumor, 1)

        fourier = self.avgpool_f(fourier)
        fourier = torch.flatten(fourier, 1)
        #x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")

        result = torch.cat((tumor,fourier), dim=1)
        result = self.fc_both(result)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return result

class NetConstant_noBN_l4_separateexperiment_fc(torch.nn.Module): #ALWAYS INCLUDES FT
    def __init__(self, block, layers, numoutputs, channels, dropoutrate):
        super(NetConstant_noBN_l4_separateexperiment_fc, self).__init__()

        self.channels = channels
        ###### SPACE FEATURE EXTRACTION ######
        self.conv1 = torch.nn.Conv3d(2, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1 = torch.nn.ReLU()

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        #self.bn_final = torch.nn.BatchNorm3d(channels)
        #self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        #self.fc = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()
        ###### SPACE FEATURE EXTRACTION ######      

        ###### FREQUENCY FEATURE EXTRACTION ######

        self.conv1_f = torch.nn.Conv3d(1, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1_f = torch.nn.ReLU()

        self.layer1_f = self._make_layer(block, layers[0], downsample=False)
        self.layer2_f = self._make_layer(block, layers[1])
        self.layer3_f = self._make_layer(block, layers[2])
        self.layer4_f = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        #self.bn_final = torch.nn.BatchNorm3d(channels)
        #self.relu_final = torch.nn.ReLU()
        self.avgpool_f = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        #self.fc_f = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()
        ###### FREQUENCY FEATURE EXTRACTION ######     

        self.fc_both = torch.nn.Linear(2*self.channels, 128)
        self.relu2 = torch.nn.ReLU()
        self.do1 = torch.nn.Dropout(p=dropoutrate)
        self.fc2 = torch.nn.Linear(128,128)
        self.relu3 = torch.nn.ReLU()
        self.do2 = torch.nn.Dropout(p=dropoutrate)
        self.fc3 = torch.nn.Linear(128, numoutputs)

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
        layers.append(block(self.channels, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.channels))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        #fourier = x[:,2,:,:,:][:,None,:,:,:] #extracts fourier transform from input
        #tumor = torch.cat((x[:,0,:,:,:][:,None,:,:,:], x[:,1,:,:,:][:,None,:,:,:]), dim=1)

        x = torch.chunk(x,2,dim=1)
        tumor = x[0]
        fourier = x[1]

        #print(fourier.shape)
        #print(tumor.shape)
        #print("finished separating input x")

        tumor = self.conv1(tumor)
        fourier = self.conv1_f(fourier)

        tumor = self.relu1(tumor)
        fourier = self.relu1_f(fourier)

        tumor = self.layer1(tumor)
        fourier = self.layer1_f(fourier)

        tumor = self.layer2(tumor)
        fourier = self.layer2_f(fourier)

        tumor = self.layer3(tumor)
        fourier = self.layer3_f(fourier)

        tumor = self.layer4(tumor)
        fourier = self.layer4_f(fourier)
        #x = self.layer5(x)

        #x = self.bn_final(x)
        #x = self.relu_final(x)

        tumor = self.avgpool(tumor)
        tumor = torch.flatten(tumor, 1)

        fourier = self.avgpool_f(fourier)
        fourier = torch.flatten(fourier, 1)
        #x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")

        result = torch.cat((tumor,fourier), dim=1)
        result = self.fc_both(result)
        result = self.relu2(result)
        result = self.do1(result)
        result = self.fc2(result)
        result = self.relu3(result)
        result = self.do2(result)
        result = self.fc3(result)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return result
'''
class ResNetInv_12_32(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv_12_32, self).__init__()

        self.inplanes = 2 #initial number of channels

        self.conv1 = torch.nn.Conv3d(self.inplanes, 32, kernel_size=7, stride=2, padding=2, bias=True)
        #self.bn1 = torch.nn.BatchNorm3d(32)
        self.relu1 = torch.nn.ReLU()

        self.inplanes = 32

        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(32, numoutputs)
        #self.tanh = torch.nn.Tanh()

        #TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                print("initializing linear")
                torch.nn.init.xavier_uniform_(m.weight)

        for m in self.modules():
            if isinstance(m, BasicBlockInv):
                torch.nn.init.constant_(m.bn2.weight, 0)
                print("initialized bn2!")
                

    def _make_layer(self, block, planes, blocks, stride):
        downsample = False
        if stride != 1 or self.inplanes != planes:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.do(x)
        #print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        #print(f"Layer after fc: {x.mean()}, {x.std()}")

        return x
'''

class NetConstant_noBN_l4_noglobalpool(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels, includesft=False, dropoutrate=0.5):
        super(NetConstant_noBN_l4_noglobalpool, self).__init__()

        if not includesft:
            self.inplanes = 2  # initial number of channels
        else:
            self.inplanes = 3

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=False)
        #self.bn1 = torch.nn.BatchNorm3d(channels)
        #self.relu1 = torch.nn.ReLU()
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        self.maxpool_final = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn_final = torch.nn.BatchNorm3d(channels)
        self.relu_final = torch.nn.ReLU()
        #self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.avgpool_final = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        #self.bn_final2 = torch.nn.BatchNorm3d(8)
        #self.relu_final2 = torch.nn.ReLU()
        #self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(4*4*4*channels, numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                #raise Exception("no batchnorm")
            #elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.xavier_uniform_(m.weight)
        #see pytorch resnet implementation cited in code before
        for m in self.modules():
            if isinstance(m, block):
                torch.nn.init.constant_(m.bn2.weight, 0)
                print("initialized bn2!")

    def _make_layer(self, block, blocks, downsample=True):
        layers = []
        layers.append(block(self.inplanes, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)

        #x = self.avgpool(x)
        x = self.maxpool_final(x)
        x = self.bn_final(x)
        x = self.relu_final(x)
        #x = self.conv_final(x)
        #x = self.bn_final2(x)
        #x = self.relu_final2(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x


class NetConstant_noBN_l4_extended(torch.nn.Module): #HERE INCLUDESFT IS DIFFERENT --> ONLY USES FT AS INPUT!!
    def __init__(self, block, layers, numoutputs, channels, includesft=False):
        super(NetConstant_noBN_l4_extended, self).__init__()

        self.includesft = includesft

        if not self.includesft:
            self.inplanes = 2  # initial number of channels
        else:
            self.inplanes = 1  # INPUT IS ONLY FOURIER TRANSFORM

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2], normalize=True)
        self.layer4 = self._make_layer(block, layers[3], normalize=True)
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
            #elif isinstance(m, torch.nn.InstanceNorm3d):
            #    torch.nn.init.constant_(m.weight, 1)
            #    torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, blocks, downsample=True, normalize=False):
        layers = []
        layers.append(block(self.inplanes, downsample, normalize))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, False, normalize))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        #if self.includesft:
        #    x = torch.chunk(x,2,dim=1)[1] #chunks [binary, pet, fourier] --> [binary,pet], [fourier] and takes fourier

        x = self.conv1(x)
        x = self.relu1(x)

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

class NetConstant_l4_extended_norm(torch.nn.Module): #HERE INCLUDESFT IS DIFFERENT --> ONLY USES FT AS INPUT!!
    def __init__(self, block, layers, numoutputs, channels, includesft=False):
        super(NetConstant_l4_extended_norm, self).__init__()

        self.includesft = includesft

        if not self.includesft:
            self.inplanes = 2  # initial number of channels
        else:
            self.inplanes = 1  # INPUT IS ONLY FOURIER TRANSFORM

        self.conv1 = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False, normalize=True)
        self.layer2 = self._make_layer(block, layers[1], normalize=True)
        self.layer3 = self._make_layer(block, layers[2], normalize=True)
        self.layer4 = self._make_layer(block, layers[3], normalize=True)
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
            #elif isinstance(m, torch.nn.InstanceNorm3d):
            #    torch.nn.init.constant_(m.weight, 1)
            #    torch.nn.init.constant_(m.bias, 0)
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, blocks, downsample=True, normalize=False):
        layers = []
        layers.append(block(self.inplanes, downsample, normalize))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, False, normalize))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        #if self.includesft:
        #    x = torch.chunk(x,2,dim=1)[1] #chunks [binary, pet, fourier] --> [binary,pet], [fourier] and takes fourier

        x = self.conv1(x)
        x = self.relu1(x)

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

def ResNetInvBasic(numoutputs, dropoutrate):
    return ResNetInv(BasicBlockInv, [3,3,4,4,2], numoutputs, dropoutrate)

def ResNetInv2Deeper(numoutputs, dropoutrate):
    return ResNetInv2(BasicBlockInv, [2,4,5,6,4,2], numoutputs, dropoutrate)

def ResNetInv2DeeperPool(numoutputs, dropoutrate):
    return ResNetInv2Pool(BasicBlockInv, [2,4,5,6,4,2], numoutputs, dropoutrate)

def ResNetInv2DeeperPoolNL(numoutputs, dropoutrate):
    return ResNetInv2PoolNL(BasicBlockInv, [2,4,5,6,4,2], numoutputs, dropoutrate)

def ResNetInv2SmallPool(numoutputs, dropoutrate):
    return ResNetInv2Pool(BasicBlockInv, [1,0,0,0,0,0], numoutputs, dropoutrate)

def ResNetInv2SmallPool_5(numoutputs, dropoutrate):
    return ResNetInv2Pool_5(BasicBlockInv, [1,0,0,0,0], numoutputs, dropoutrate)

def ResNetInvPreActDirect_Small(numoutputs, dropoutrate): #like resnet18
    return ResNetInvPreActDirect(BasicBlockInv_PreAct_Pool, [2,2,2,2,2], numoutputs, dropoutrate)

def ResNetInvPreActDirect_Medium(numoutputs, dropoutrate): #like resnet34
    return ResNetInvPreActDirect(BasicBlockInv_PreAct_Pool, [2,3,4,6,3], numoutputs, dropoutrate)

def ResNetInvPreActDirect_Wider_2_Medium(numoutputs, dropoutrate):
    return ResNetInvPreActDirect_Wider_2(BasicBlockInv_PreAct_Pool, BasicBlockInv_PreAct_Pool_constant, [1,6,2,1,1], numoutputs, dropoutrate, 16)

def PreActNetConstant_16_n1(numoutputs, dropoutrate): #we keep dropout rate although unused so don't change main.py code
    return PreActNetConstant(BasicBlockInv_PreAct_Pool_constant, [1,1,1,1,1], numoutputs, 16)

def NetConstant_noBN_16_n1(numoutputs, dropoutrate): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN(BasicBlockInv_Pool_constant_noBN, [1,1,1,1,1], numoutputs, 16)

def NetConstant_noBN_32_n1(numoutputs, dropoutrate): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN(BasicBlockInv_Pool_constant_noBN, [1,1,1,1,1], numoutputs, 32)

def NetConstant_noBN_64_n1(numoutputs, dropoutrate): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN(BasicBlockInv_Pool_constant_noBN, [1,1,1,1,1], numoutputs, 64)

def NetConstant_noBN_64_n1_l4(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN_l4(BasicBlockInv_Pool_constant_noBN, [1,1,1,1], numoutputs, 64, includesft=includesft)

def NetConstant_noBN_16_n1_l4(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN_l4(BasicBlockInv_Pool_constant_noBN, [1,1,1,1], numoutputs, 16, includesft=includesft)

def NetConstant_noBN_16_n1_l4_TWONET(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    assert includesft == True
    return NetConstant_noBN_l4_separateexperiment(BasicBlockInv_Pool_constant_noBN, [1,1,1,1], numoutputs, 16)

def NetConstant_noBN_16_n1_l4_TWONET_fc(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    assert includesft == True
    return NetConstant_noBN_l4_separateexperiment_fc(BasicBlockInv_Pool_constant_noBN, [1,1,1,1], numoutputs, 16, dropoutrate)

def NetConstant_noBN_32_n2_l4(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN_l4(BasicBlockInv_Pool_constant_noBN, [2,2,2,2], numoutputs, 32, includesft=includesft)

def ResNetInv_12_f32(numoutputs, dropoutrate):
    return ResNetInv_12_32(BasicBlockInv, [2,2,2,2], numoutputs, dropoutrate)

def NetConstant_64_n2_l4_noglobalpool(numoutputs, dropoutrate, includesft):
    return NetConstant_noBN_l4_noglobalpool(BasicBlockInv_PreAct_Pool_constant, [2,2,2,2], numoutputs, 64, includesft, dropoutrate)

def NetConstant_noBN_64_n4_l4(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN_l4(BasicBlockInv_Pool_constant_noBN_n4, [1,1,1,1], numoutputs, 64, includesft=includesft)

def NetConstant_IN_normtail(numoutputs,dropoutrate,includesft):
    return NetConstant_noBN_l4_extended(BasicBlockInv_Pool_constant_n4_inorm, [2,2,2,2], numoutputs, 64, includesft=includesft)

def NetConstant_IN_norm(numoutputs,dropoutrate,includesft):
    return NetConstant_l4_extended_norm(BasicBlockInv_Pool_constant_n4_inorm, [2,2,2,2], numoutputs, 64, includesft=includesft)

def save_inverse_model(savelogdir, epoch, model_state_dict, optimizer_state_dict, best_val_loss, total_train_loss,
                       dropoutrate, batch_size, numoutputs, learning_rate, lr_scheduler_rate,
                       starttrain, endtrain, startval, endval, version, schedulername, lossfunctionname, seed, is_debug,
                       optimizername, weight_decay_sgd, modelfun_name, lr_patience, includesft, outputmode,
                       passedarguments,
                       summarystring, additionalsummary, savefilename):
    if not is_debug:
        torch.save({'epoch': epoch, 'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer_state_dict,
                    'best_val_loss': best_val_loss, 'total_train_loss': total_train_loss, 'dropoutrate': dropoutrate,
                    'batch_size': batch_size, 'numoutputs': numoutputs, 'learning_rate': learning_rate,
                    'lr_scheduler_rate': lr_scheduler_rate, 'starttrain': starttrain, 'endtrain': endtrain,
                    'startval': startval, 'endval': endval, 'version': version, 'schedulername': schedulername,
                    'lossfunctionname': lossfunctionname, 'seed': seed, 'modelfun_name': modelfun_name,
                    'optimizername': optimizername, 'weight_decay_sgd': weight_decay_sgd, 'lr_patience': lr_patience,
                    'includesft': includesft, 'outputmode': outputmode, 'passedarguments': passedarguments,
                    'summarystring': summarystring, 'additionalsummary': additionalsummary},
                   savelogdir + savefilename)

def load_inverse_model(loaddir):
    return torch.load(loaddir)


def ranged_error(name, index, original_range, ys, yspredicted, observed_range, loaddir):
    var_normalized = ys[:, index]
    vars = np.interp(var_normalized, normalization_range, original_range)

    predicted_var_normalized = yspredicted[:, index]
    predicted_vars = np.interp(predicted_var_normalized, normalization_range,
                                  original_range)

    vars = np.interp(vars, observed_range, [0.0, 1.0])
    predicted_vars = np.interp(predicted_vars, observed_range, [0.0, 1.0])

    ranged_error_var = np.abs(predicted_vars - vars)

    mean = np.mean(ranged_error_var)
    std = np.std(ranged_error_var)

    print("Ranged " + name + " error: mean = " + str(np.round(mean, 6)) + ", std = " + str(np.round(std, 6)))

    plt.figure()
    plt.hist(ranged_error_var, bins=100, range=(0.0, 1.0))
    plt.title(loaddir + ": \n " + "ranged error " + name)
    plt.savefig("fig" + name + ".png")

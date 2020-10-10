import torch
import numpy as np
import pickle5 as pickle
import sys
import os
from glob import glob
#e.g. Dataset("/home/kevin/Desktop/thresholding/", 0, 9)
normalization_range = [-1.0, 1.0]

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datapath, beginning, ending):
        'Initialization'
        #beginning inclusive, ending exclusive
        self.datapath = datapath
        self.beginning = beginning
        self.ending = ending
        
        self.all_paths = sorted(glob("{}/*/".format(self.datapath)))[self.beginning : self.ending]

  def __len__(self):
        'Denotes the total number of samples'
        #return (self.ending - self.beginning)
        return len(self.all_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        file_path = self.all_paths[index]
        #print("got pos " + str(index) + " which corresponds to " + str(file_path))
        with np.load(file_path + "Data_0001.npz") as data:
            #thrvolume = data['thr_data']
            thrvolume = data['data']
            thrvolume_resized = np.delete(np.delete(np.delete(thrvolume, 128, 0), 128, 1), 128, 2) #from 129x129x129 to 128x128x128
            #TODO: check if deletion removed nonzero entries (especially last slice: thrvolume[...][...][128])
            thrvolume_resized = np.expand_dims(thrvolume_resized, -1) #now it is 128x128x128x1

        with open(file_path + "parameter_tag.pkl", "rb") as par:
            #TODO: interpolate with manual formulas (e.g. uth: 10x - 7)
            #TODO: rounding to 6 digits?
            paramsarray = np.zeros(3)
            params = pickle.load(par)
            #paramsarray[0] = np.interp(params['uth'], [0.6, 0.8], normalization_range) #TODO: change range -> still uses [0.6, 0.8] range!!
            #paramsarray[0] = np.interp(params['Dw'], [0.0002, 0.015], normalization_range)
            #paramsarray[1] = np.interp(params['rho'], [0.002, 0.2], normalization_range)
            #paramsarray[2] = np.interp(params['Tend'], [50, 1500], normalization_range)
            paramsarray[0] = np.interp(params['icx'], [0.15, 0.7], normalization_range)
            paramsarray[1] = np.interp(params['icy'], [0.2, 0.8], normalization_range)
            paramsarray[2] = np.interp(params['icz'], [0.15, 0.7], normalization_range)

        thrvolume_resized = thrvolume_resized.transpose((3,0,1,2))
        return torch.from_numpy(thrvolume_resized.astype(np.float32)), torch.from_numpy(paramsarray.astype(np.float32))

##########################################################################################################


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)

def conv1x1(in_planes, out_planes, stride=1, padding=0):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding)

class ConvNet(torch.nn.Module):

    def __init__(self, numoutputs):
        super(ConvNet, self).__init__()

        self.seq = torch.nn.Sequential(
                    conv3x3(1,2),
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
                    torch.nn.Dropout(p=0.2),
                    torch.nn.Linear(8*8*8*32, 3)
                    )

    def forward(self, x):
        return self.seq(x)

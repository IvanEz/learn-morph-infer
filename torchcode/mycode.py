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
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1, padding=0):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)

class ConvNet(torch.nn.Module):

    def __init__(self, numoutputs, dropoutrate=0.2):
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
                    torch.nn.Dropout(p=dropoutrate),
                    torch.nn.Linear(8*8*8*32, numoutputs)
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
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample
        if self.downsample:
            self.conv3 = conv1x1(inplanes, planes, stride)
            self.bn3 = torch.nn.BatchNorm3d(planes)

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is True:
            identity = self.conv3(identity)
            identity = self.bn3(identity)

        out += identity
        out = self.relu2(out)

        return out

class ResNetInv(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv, self).__init__()

        self.inplanes = 1 #initial number of channels

        self.layer1 = self._make_layer(block, 1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(8*8*8*32, numoutputs)

        #TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias, 0)

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
        x = self.fc(x)

        return x

class ResNetInv2(torch.nn.Module):

    def __init__(self, block, layers, numoutputs, dropoutrate):
        super(ResNetInv, self).__init__()

        self.inplanes = 1 #initial number of channels

        self.relu = torch.nn.ReLU()
        self.layer1 = self._make_layer(block, 1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 4, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 8, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 32, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 64, layers[5], stride=2)
        self.fc = torch.nn.Linear(4*4*4*64, 64)
        self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc2 = torch.nn.Linear(64, 64)
        self.do2 = torch.nn.Dropout(p=dropoutrate)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 7)

        #TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                torch.nn.BatchNorm3d(planes)
            )

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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.do2(x)
        x = self.fc3(x)
        x = self.relu(x)
        #here dropout maybe
        x = self.fc4(x)

        return x

def ResNetInvBasic(numoutputs, dropoutrate):
    return ResNetInv(BasicBlockInv, [3,3,4,4,2], numoutputs, dropoutrate)

def save_inverse_model(savelogdir, epoch, model_state_dict, optimizer_state_dict, best_val_loss, total_train_loss,
                       dropoutrate, batch_size, numoutputs, learning_rate,
                       summarystring, additionalsummary):
    torch.save({'epoch': epoch, 'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer_state_dict,
                'best_val_loss': best_val_loss, 'total_train_loss': total_train_loss, 'dropoutrate': dropoutrate,
                'batch_size': batch_size, 'numoutputs': numoutputs, 'learning_rate': learning_rate,
                'summarystring': summarystring, 'additionalsummary': additionalsummary},
               savelogdir + '/bestval-model.pt')

def load_inverse_model(loaddir):
    return torch.load(loaddir)
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

        self.inplanes = 1 #initial number of channels

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
        super(ResNetInv2, self).__init__()

        self.inplanes = 1  # initial number of channels

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

def ResNetInvBasic(numoutputs, dropoutrate):
    return ResNetInv(BasicBlockInv, [3,3,4,4,2], numoutputs, dropoutrate)

def ResNetInv2Deeper(numoutputs, dropoutrate):
    return ResNetInv2(BasicBlockInv, [2,4,5,6,4,2], numoutputs, dropoutrate)

def ResNetInv2DeeperPool(numoutputs, dropoutrate):
    return ResNetInv2Pool(BasicBlockInv, [2,4,5,6,4,2], numoutputs, dropoutrate)

def save_inverse_model(savelogdir, epoch, model_state_dict, optimizer_state_dict, best_val_loss, total_train_loss,
                       dropoutrate, batch_size, numoutputs, learning_rate, lr_scheduler_rate,
                       starttrain, endtrain, startval, endval, version, schedulername, lossfunctionname,
                       summarystring, additionalsummary, savefilename):
    torch.save({'epoch': epoch, 'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer_state_dict,
                'best_val_loss': best_val_loss, 'total_train_loss': total_train_loss, 'dropoutrate': dropoutrate,
                'batch_size': batch_size, 'numoutputs': numoutputs, 'learning_rate': learning_rate,
                'lr_scheduler_rate': lr_scheduler_rate, 'starttrain': starttrain, 'endtrain': endtrain,
                'startval': startval, 'endval': endval, 'version': version, 'schedulername': schedulername,
                'lossfunctionname': lossfunctionname,
                'summarystring': summarystring, 'additionalsummary': additionalsummary},
               savelogdir + savefilename)

def load_inverse_model(loaddir):
    return torch.load(loaddir)

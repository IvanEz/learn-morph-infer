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
    def __init__(self, datapath, beginning, ending, thrpath, necroticpath, num_thresholds=100, includesft=False, outputmode=0):
        Dataset.__init__(self, datapath, beginning, ending, thrpath, num_thresholds=num_thresholds)
        self.includesft = includesft
        self.outputmode = outputmode
        self.necroticpath = necroticpath
        self.necrotic_paths = sorted(glob("{}/*".format(self.necroticpath)))[self.beginning : self.ending]
        assert len(self.necrotic_paths) == self.datasetsize

    def __len__(self):
        return self.datasetsize

    def __getitem__(self, index):
        file_path = self.all_paths[index]
        thr_path = self.threshold_paths[index]
        necrotic_path = self.necrotic_paths[index]

        with np.load(thr_path) as thresholdsfile:
            t1gd_thr = thresholdsfile['t1gd'][self.epoch % self.num_thresholds]
            flair_thr = thresholdsfile['flair'][self.epoch % self.num_thresholds]
            assert t1gd_thr >= 0.5 and t1gd_thr <= 0.85
            assert flair_thr >= 0.05 and flair_thr <= 0.5

        with np.load(necrotic_path) as necroticfile:
            necrotic_thr = necroticfile['necrotic'][self.epoch % self.num_thresholds]
            assert necrotic_thr >= 0.95 and necrotic_thr <= 1.0
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

            #b = 0.5

            pet_volume = ((volume_resized >= t1gd_thr) * volume_resized)
            pet_volume = (pet_volume <= necrotic_thr) * pet_volume
            pet_volume_max = pet_volume.max()
            assert pet_volume_max >= 0.0
            if pet_volume_max == 0.0:
                print(f"LIGHT WARNING: empty pet volume for {file_path}")
                #no division by max, volume is left empty
            else:
                pet_volume = pet_volume / pet_volume.max()
            #print(pet_volume.shape)
            pet_volume_reshaped = np.expand_dims(pet_volume, -1) #now 129x129x129x1
            #print(pet_volume_reshaped.shape)

            nn_input = np.concatenate((thrvolume_resized, pet_volume_reshaped), -1)
            #print(nn_input.shape)

            if self.includesft:
                '''
                ft = np.abs(np.fft.fftshift(np.fft.fftn(thr_volume + pet_volume, norm='ortho')))
                ft_reshaped = np.expand_dims((ft / np.max(ft)), -1)
                #nn_input = np.concatenate((nn_input, ft_reshaped), -1)
                nn_input = ft_reshaped  # OVERWRITES NN_INPUT, IS NOW ONLY FOURIER TRANSFORM, NOT SPATIAL TUMOR!
                if index == 0:
                    print("Shape is " + str(nn_input.shape) + ", should be (129,129,129,1)")
                '''
                raise Exception("no support for ft in this version")

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
            elif self.outputmode == 4:
                paramsarray = np.zeros(2)
                sqrtDT = np.sqrt(Dw * Tend)
                sqrtmu = np.sqrt(mu)
                paramsarray[0] = np.interp(sqrtDT, [0.1, np.sqrt(22.5)], normalization_range)
                paramsarray[1] = np.interp(sqrtmu, np.sqrt([0.1, 300.0]), normalization_range)
            elif self.outputmode == 5:
                paramsarray = np.zeros(3)
                paramsarray[0] = np.interp(params['Dw'], [0.0002, 0.015], normalization_range)
                paramsarray[1] = np.interp(params['rho'], [0.002, 0.2], normalization_range)
                paramsarray[2] = np.interp(params['Tend'], [50, 1500], normalization_range)
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

#adapted from https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html and https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
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

class NetConstant_noBN_l4_inplacefull(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels, includesft=False):
        super(NetConstant_noBN_l4_inplacefull, self).__init__()

        if not includesft:
            self.inplanes = 2  # initial number of channels
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

def NetConstant_noBN_64_n4_l4_inplace(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN_l4(BasicBlockInv_Pool_constant_noBN_n4_inplace, [1,1,1,1], numoutputs, 64, includesft=includesft)

def NetConstant_noBN_64_n4_l4_inplacefull(numoutputs, dropoutrate, includesft): #we keep dropout rate although unused so don't change main.py code
    return NetConstant_noBN_l4_inplacefull(BasicBlockInv_Pool_constant_noBN_n4_inplace, [1,1,1,1], numoutputs, 64, includesft=includesft)

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

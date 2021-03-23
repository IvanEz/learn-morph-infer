
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
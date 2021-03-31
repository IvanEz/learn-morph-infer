#main.py and dataloader.py are based on: https://github.com/rasbt/deeplearning-models of Sebastian Raschka, https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel, https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py, https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.mdtor, https://github.com/pytorch/examples/blob/master/imagenet/main.py, PyTorch Tutorials on pytorch.org/tutorials
#torchsummary.py has been written by sksq96

import time
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle5 as pickle
import os
from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary_string
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from dataloader import *
from network import *
from train import *
from test import *

seed = np.random.randint(0,10000000)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

version = "v7_1" #includes necrotic core + normalized pet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=4, type=int)
parser.add_argument('--isnewsave', action='store_true', help="if true -> training, if not --> inference")
parser.add_argument('--isdebug', action='store_true', help="saves statistics but does not create .pt files")
parser.add_argument('--purpose', default="resnetdeep-80000-dropout2-lr0001-exp978", type=str, help="the net type, if isnewsave: include!")
parser.add_argument('--loaddir', default="", type=str, help="path to trained model, if is not new save: include!")
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--dropoutrate', default=0.2, type=float)
parser.add_argument('--lr_scheduler_rate', default=0.9999, type=float)
parser.add_argument('--starttrain', default=0, type=int)
parser.add_argument('--endtrain', default=80000, type=int)
parser.add_argument('--startval', default=80000, type=int)
parser.add_argument('--endval', default=88000, type=int)
parser.add_argument("--num_thresholds", default=100, type=int, choices=range(1,101))
parser.add_argument('--is_sgd', action='store_true')
parser.add_argument('--reduce_lr_on_flat', action='store_true')
parser.add_argument('--weight_decay_sgd', default=0.0, type=float, help="weight decay for adam AND sgd (needs to be renamed)")
parser.add_argument('--lr_patience', default=10, type=int)
parser.add_argument('--includesft', action='store_true', help="include or not fourier transform")
parser.add_argument('--outputmode', default=0, type=int)
parser.add_argument('--savelogdir', default="./result/", type=str)


#outputmode defines the outputs of the network
#0: uth1, uth2, lambda, mu, v, x, y, z (as is in latest versions)
#1: lambda,mu,v
#2: x,y,z
#3: lambda, mu
#6: lambda, mu,x,y,z

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

##########################
### SETTINGS
##########################

is_new_save = args.isnewsave
is_debug = args.isdebug
loaddir = args.loaddir #choose directory from which to load from if is_new_save = False, do not end with '/'

if not is_new_save:
    checkpoint = load_inverse_model(loaddir)

# Experiment specification
currenttime = datetime.now()
currenttime = currenttime.strftime("%d%m-%H-%M-%S-")
purpose = version + "-" + args.purpose

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = args.lr #0.0001 was good
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
dropoutrate = args.dropoutrate if is_new_save else checkpoint['dropoutrate']
lr_scheduler_rate = args.lr_scheduler_rate
num_thresholds = args.num_thresholds

is_sgd = args.is_sgd
reduce_lr_on_flat = args.reduce_lr_on_flat
weight_decay_sgd = args.weight_decay_sgd
lr_patience = args.lr_patience

includesft = args.includesft

if is_new_save:
    outputmode = args.outputmode
else:
    outputmode = checkpoint['outputmode']
print("OUTPUT MODE: {outputmode}")

# Defining the size of the output, see line 177 in dataloader.py
if is_new_save:
    if outputmode == 0:
        numoutputs = 8
    elif outputmode == 1 or outputmode == 2:
        numoutputs = 3
    elif outputmode == 3:
        numoutputs = 2
    elif outputmode == 4:
        numoutputs = 2
    elif outputmode == 5:
        numoutputs = 3
    else:
        raise Exception("invalid output mode")
else:
    numoutputs = checkpoint['numoutputs']

# Dataloading
starttrain = args.starttrain
endtrain = args.endtrain #6400 / 16000 / 32000 / 64000 / 80000
startval = args.startval #6400 / 16000 / 32000 / 64000 / 80000 - external validation: 80000
endval = args.endval #7040 / 17600 - 17664 / 35200 / 70400 / 88000 - external validation: 88000

train_dataset = Dataset2("/home/ivan/ib/learnmorph/samples_extended_thr2/Dataset/", starttrain, endtrain,
                        "/home/ivan/ib/learnmorph/files", "/home/ivan/ib/learnmorph/necroticthrs",
                         num_thresholds=num_thresholds, includesft=includesft,
                         outputmode=outputmode)
train_generator = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = Dataset2("/home/ivan/ib/learnmorph/samples_extended_thr2/Dataset/", startval, endval,
                      "/home/ivan/ib/learnmorph/files", "/home/ivan/ib/learnmorph/necroticthrs",
                       includesft=includesft, outputmode=outputmode)
val_generator = torch.utils.data.DataLoader(val_dataset, 
                    batch_size=batch_size, shuffle=False, num_workers=num_workers)

if is_new_save:
    assert len(train_dataset) % batch_size == 0
    if len(val_dataset) % batch_size != 0:
        print("WARNING: val dataset size is not multiple of batch size!")
        time.sleep(30)

# Setting up model
if is_new_save:
    savelogdir = '/home/ivan/ib/learnmorph/log/torchimpl/' + currenttime + purpose
    writer = SummaryWriter(log_dir = savelogdir)
    writerval = SummaryWriter(log_dir = savelogdir + '/val')

modelfun = NetConstant_noBN_64_n4_l4_inplacefull
model = modelfun(numoutputs=numoutputs, dropoutrate=dropoutrate, includesft=includesft)
modelfun_name = modelfun.__name__

##### Summaries #####
summarystring = repr(model)
print(summarystring)

if not includesft:
    additionalsummary, _ = summary_string(model, (2,129,129,129), device="cpu") #additional summary is done on cpu (only once), model not yet on gpu
else:
    additionalsummary, _ = summary_string(model, (1, 129, 129, 129), device="cpu")
print(additionalsummary)

if is_new_save:
    if not includesft:
        writer.add_graph(model, Variable(torch.rand(1,2,129,129,129)))
    else:
        writer.add_graph(model, Variable(torch.rand(1, 1, 129, 129, 129)))

##### Sending model to gpu, and defining oprtimizer
if not is_new_save:
    model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

if not is_sgd:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_sgd)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                                weight_decay=weight_decay_sgd, nesterov=True) #we use nesterov, CS231n recommendation
    print("WARNING: you are using SGD")
    time.sleep(20)

optimizername = optimizer.__class__.__name__

if not is_new_save:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#################
### Training
#################
step = 0 #1 step = 1 pass through 1 batch
step_val = 0
best_val_loss = 999999.0

if not reduce_lr_on_flat:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_scheduler_rate) #0.95 - 0.978
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience)
schedulername = scheduler.__class__.__name__

numberofbatches = len(train_generator)
numberofbatches_val = len(val_generator)

#lossfunction = F.l1_loss
lossfunction = F.mse_loss
lossfunctionname = lossfunction.__name__

if is_new_save:
    train(num_epochs, model, train_dataset, train_generator, optimizer, device, lossfunction, writer,
              numberofbatches, numberofbatches_val,step,
              reduce_lr_on_flat, scheduler, val_generator, step_val, best_val_loss, savelogdir, writerval,
              dropoutrate, batch_size, numoutputs, learning_rate, lr_scheduler_rate, starttrain, endtrain, startval,
              endval, version, schedulername, lossfunctionname, seed, is_debug, optimizername, weight_decay_sgd,
              modelfun_name, lr_patience, includesft, outputmode, args, summarystring, additionalsummary
              )
else:
    test(num_epochs, model, train_dataset, train_generator, optimizer, device, lossfunction, writer,
             numberofbatches, numberofbatches_val, step,
             reduce_lr_on_flat, scheduler, val_generator, step_val, best_val_loss, savelogdir, writerval,
             dropoutrate, batch_size, numoutputs, learning_rate, lr_scheduler_rate, starttrain, endtrain, startval,
             endval, version, schedulername, lossfunctionname, seed, is_debug, optimizername, weight_decay_sgd,
             modelfun_name, lr_patience, includesft, outputmode, args, summarystring, additionalsummary
             )

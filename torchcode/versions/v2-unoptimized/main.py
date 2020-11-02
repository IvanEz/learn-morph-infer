#main.py and mycode.py are based on: https://github.com/rasbt/deeplearning-models of Sebastian Raschka, https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel, https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py, https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.mdtor, https://github.com/pytorch/examples/blob/master/imagenet/main.py, PyTorch Tutorials on pytorch.org/tutorials

#torchsummary.py has been written by sksq96

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from mycode import *
import matplotlib.pyplot as plt

from torchsummary import summary_string
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse

version = "v2"

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=4, type=int)
parser.add_argument('--isnewsave', action='store_true') #--isnewsave -> training, if not there: inference

parser.add_argument('--purpose', default="resnetdeep-80000-dropout2-lr0001-exp978", type=str) #if isnewsave: include!
parser.add_argument('--loaddir', default="/mnt/Drive2/ivan_kevin/log/torchimpl/2410-08-01-59-resnetdeep-64000-dropout4-lr0001-exp978/bestval-model.pt", type=str) #if is not new save: include!

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

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

#if torch.cuda.is_available():
#    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

is_new_save = args.isnewsave
loaddir = args.loaddir #choose directory from which to load from if is_new_save = False, do not end with '/'

if not is_new_save:
    checkpoint = load_inverse_model(loaddir)

# Experiment specification
currenttime = datetime.now()
currenttime = currenttime.strftime("%d%m-%H-%M-%S-")
purpose = version + "-" + args.purpose
#purpose = "resnetdeep-32000-dropout4-lr0005-exp75"


# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
#random_seed = 123
learning_rate = args.lr #0.0001 was good
num_epochs = args.num_epochs
batch_size = args.batch_size if is_new_save else checkpoint['batch_size']
num_workers = args.num_workers
dropoutrate = args.dropoutrate if is_new_save else checkpoint['dropoutrate']
lr_scheduler_rate = args.lr_scheduler_rate
num_thresholds = args.num_thresholds

# Architecture
numoutputs = 8 if is_new_save else checkpoint['numoutputs']


starttrain = args.starttrain
endtrain = args.endtrain #6400 / 16000 / 32000 / 64000 / 80000
startval = args.startval #6400 / 16000 / 32000 / 64000 / 80000 - external validation: 80000
endval = args.endval #7040 / 17600 - 17664 / 35200 / 70400 / 88000 - external validation: 88000

train_dataset = Dataset("/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/", starttrain, endtrain,
                        "/mnt/Drive2/ivan_kevin/thresholds/files", num_thresholds=num_thresholds)
train_generator = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = Dataset("/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/", startval, endval,
                      "/mnt/Drive2/ivan_kevin/thresholds/files")
val_generator = torch.utils.data.DataLoader(val_dataset,
                    batch_size=batch_size, shuffle=False, num_workers=num_workers)

assert len(train_dataset) % batch_size == 0
assert len(val_dataset) % batch_size == 0

# Setting up model
if is_new_save:
    savelogdir = '/mnt/Drive2/ivan_kevin/log/torchimpl/' + currenttime + purpose
    writer = SummaryWriter(log_dir = savelogdir)
    writerval = SummaryWriter(log_dir = savelogdir + '/val')

#torch.manual_seed(random_seed)
model = ResNetInv2Deeper(numoutputs=numoutputs, dropoutrate=dropoutrate)
#model = ConvNet(numoutputs=numoutputs, dropoutrate=dropoutrate)

##### summaries #####
summarystring = repr(model)
print(summarystring)

additionalsummary, _ = summary_string(model, (1,128,128,128), device="cpu") #additional summary is done on cpu (only once), model not yet on gpu
print(additionalsummary)

if is_new_save:
    writer.add_graph(model, Variable(torch.rand(1,1,128,128,128)))
##### summaries #####
##########################################
if not is_new_save:
    model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if not is_new_save:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
###########################################
# Training
step = 0 #1 step = 1 pass through 1 batch
step_val = 0

best_val_loss = 999999.0

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_scheduler_rate) #0.95 - 0.978
schedulername = scheduler.__class__.__name__ #save it in case you change it and try something new

numberofbatches = len(train_generator)
numberofbatches_val = len(val_generator)

lossfunction = F.l1_loss
lossfunctionname = lossfunction.__name__

if is_new_save:
    lastsavetime = time.time()
    for epoch in range(num_epochs):
        #training
        train_dataset.epoch = epoch
        running_training_loss = 0.0
        model = model.train()
        for batch_idx, (x,y) in enumerate(train_generator):
            #x: volume
            #y: parameters
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            y_predicted = model(x)
            cost = lossfunction(y_predicted, y)
            #cost = F.mse_loss(y_predicted, y)
            cost.backward()

            optimizer.step()


            running_training_loss += cost
            writer.add_scalar('Loss/train', cost, step)

            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.6f'
                   %(epoch+1, num_epochs, batch_idx + 1,
                     numberofbatches, cost))

            step += 1



        total_train_loss = (running_training_loss / numberofbatches).item() #avg loss in epoch!
        print('Epoch: %03d | Average loss: %.6f'%(epoch+1, total_train_loss))
        writer.add_scalar('Loss/avg_train', total_train_loss, epoch)
        writer.add_scalar('losses', total_train_loss, epoch)

        #validation
        running_validation_loss = 0.0
        model = model.eval()
        with torch.set_grad_enabled(False):
            for batch_idy, (x,y) in enumerate(val_generator):
                x, y = x.to(device), y.to(device)
                y_predicted = model(x)
                cost = lossfunction(y_predicted, y)
                #cost = F.mse_loss(y_predicted, y)

                running_validation_loss += cost
                writer.add_scalar('Loss/val', cost, step_val)
                print ('Validating: %03d | Batch %03d/%03d | Cost: %.4f'
                   %(epoch+1, batch_idy + 1, numberofbatches_val, cost))
                step_val += 1

        total_val_loss = (running_validation_loss / numberofbatches_val).item()
        print('Epoch: %03d | Validation loss: %.6f'%(epoch+1, total_val_loss))
        writer.add_scalar('Loss/avg_val', total_val_loss, epoch)
        writerval.add_scalar('losses', total_val_loss, epoch)

        if np.round(total_val_loss,5) < np.round(best_val_loss,5):
            best_val_loss = total_val_loss
            save_inverse_model(savelogdir, epoch, model.state_dict(), optimizer.state_dict(), best_val_loss,
                                total_train_loss, dropoutrate, batch_size, numoutputs, learning_rate,
                                lr_scheduler_rate, starttrain, endtrain, startval, endval, version,
                                schedulername, lossfunctionname,
                               summarystring, additionalsummary, '/bestval-model.pt')
            print(">>> Saving new model with new best val loss " + str(best_val_loss))

        if (time.time() - lastsavetime) > 3600.0: #more than an hour has passed since last save
            save_inverse_model(savelogdir, epoch, model.state_dict(), optimizer.state_dict(), best_val_loss,
                               total_train_loss, dropoutrate, batch_size, numoutputs, learning_rate,
                               lr_scheduler_rate, starttrain, endtrain, startval, endval, version,
                               schedulername, lossfunctionname,
                               summarystring, additionalsummary, '/epoch' + str(epoch) + '.pt')
            lastsavetime = time.time()
            print(">>> Saved!")

        scheduler.step()
        print(scheduler.get_last_lr())
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    print("Best val loss: %.6f"%(best_val_loss))
    writer.close()
    writerval.close()

else:
    model = model.eval()

    lossmatrix = []
    ys = []
    yspredicted = []
    with torch.set_grad_enabled(False):
        for batch_idy, (x, y) in enumerate(val_generator):
            x, y = x.to(device), y.to(device)
            y_predicted = model(x)
            cost = F.l1_loss(y_predicted, y, reduction='none')
            cost = cost.cpu().numpy()
            #print(cost)
            cost = cost / 2 #takes into account that normalization range is [-1,1]
            #print(cost)
            assert cost.shape[1] == numoutputs
            lossmatrix.append(cost)
            ys.append(y.cpu().numpy())
            yspredicted.append(y_predicted.cpu().numpy())

        lossmatrix = np.concatenate(lossmatrix)
        ys = np.concatenate(ys)
        yspredicted = np.concatenate(yspredicted)

        print("##########################################")
        for outputfeature in range(numoutputs):
            losses = lossmatrix[:, outputfeature]
            print(losses.shape)
            mean = np.mean(losses)
            std = np.std(losses)

            print("Output " + str(outputfeature) + " error: mean = " + str(np.round(mean, 4)) + ", std = " + str(np.round(std, 4)))

            plt.figure()
            plt.hist(losses, bins=100)
            plt.title(loaddir + ": \n " + str(outputfeature))
            plt.savefig("fig" + str(outputfeature) + ".png")

        #print("Error based on normalization range [-1.0, 1.0]! If error output here is e.g. 0.05 --> error is 2.5% !")
        print("Best val loss was " + str(np.round(checkpoint['best_val_loss'], 6))
              + ", train loss was " + str(np.round(checkpoint['total_train_loss'],6)))
        # Mean error: what is the mean L1 loss across all samples in the validation set?
        # Std: how much do the losses deviate from the above mean in the validation set?
        # (Low: The mean error reported above for that feature is roughly that )
        print("##########################################")

        np.savez_compressed("results.npz", losses=lossmatrix, ys = ys, yspredicted = yspredicted) #still in normalization range [-1,1] !

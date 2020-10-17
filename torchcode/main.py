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

from torchsummary import summary_string
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="6"

#if torch.cuda.is_available():
#    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

is_new_save = True
loaddir = "/mnt/Drive2/ivan_kevin/log/torchimpl/1010-19-55-32-debug" #choose directory from which to load from if is_new_save = False, do not end with '/'

if not is_new_save:
    checkpoint = load_inverse_model(loaddir + "/bestval-model.pt")

# Experiment specification
currenttime = datetime.now()
currenttime = currenttime.strftime("%d%m-%H-%M-%S-")
purpose = "RESNET-debug"


# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
#random_seed = 123
learning_rate = 0.001 #0.00001 was good
num_epochs = 300
batch_size = 128 if is_new_save else checkpoint['batch_size']
num_workers = 32
dropoutrate = 0.4 if is_new_save else checkpoint['dropoutrate']

# Architecture
numoutputs = 3 if is_new_save else checkpoint['numoutputs']


starttrain = 0
endtrain = 4600
startval = 4600
endval = 5112
train_dataset = Dataset("/mnt/Drive2/ivan_kevin/samples_extended_copy/Dataset/", starttrain, endtrain)
train_generator = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = Dataset("/mnt/Drive2/ivan_kevin/samples_extended_copy/Dataset/", startval, endval)
val_generator = torch.utils.data.DataLoader(val_dataset, 
                    batch_size=batch_size, shuffle=False, num_workers=num_workers)


# Setting up model
if is_new_save:
    savelogdir = '/mnt/Drive2/ivan_kevin/log/torchimpl/' + currenttime + purpose
    writer = SummaryWriter(log_dir = savelogdir)
    writerval = SummaryWriter(log_dir = savelogdir + '/val')

#torch.manual_seed(random_seed)
model = ResNetInvBasic(numoutputs=numoutputs, dropoutrate=dropoutrate)

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

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #eps=0.1 works too, prefer lrdecay

if not is_new_save:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
###########################################
# Training
step = 0 #1 step = 1 pass through 1 batch
step_val = 0

best_val_loss = 999999.0

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

lossfunction = F.l1_loss

if is_new_save:
    for epoch in range(num_epochs):
        #training
        running_training_loss = 0.0
        model = model.train()
        for batch_idx, (x,y) in enumerate(train_generator):
            #x: volume
            #y: parameters
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            y_predicted = model(x)
            cost = lossfunction(y_predicted, y)

            cost.backward()

            optimizer.step()

            beforeitem = time.time()
            costval = cost.item()
            elapsedtimeforitem = time.time() - beforeitem

            print("Time to call item(): " + str(elapsedtimeforitem))
            running_training_loss += costval
            writer.add_scalar('Loss/train', costval, step)
            if not batch_idx % 2:
                print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.6f'
                       %(epoch+1, num_epochs, batch_idx + 1,
                         len(train_generator), costval))
            step += 1



        total_train_loss = (running_training_loss / len(train_generator)) #avg loss in epoch!
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
                costval = cost.item()

                running_validation_loss += costval
                writer.add_scalar('Loss/val', costval, step_val)
                if not batch_idy % 2:
                    print ('Validating: %03d | Batch %03d/%03d | Cost: %.4f'
                       %(epoch+1, batch_idy + 1, len(val_generator), costval))
                step_val += 1

        total_val_loss = (running_validation_loss / len(val_generator))
        print('Epoch: %03d | Validation loss: %.6f'%(epoch+1, total_val_loss))
        writer.add_scalar('Loss/avg_val', total_val_loss, epoch)
        writerval.add_scalar('losses', total_val_loss, epoch)

        if np.round(total_val_loss,2) < np.round(best_val_loss,2): #TODO: also save every x epochs --> might decrease slowly but still overfit
            best_val_loss = total_val_loss
            save_inverse_model(savelogdir, epoch, model.state_dict(), optimizer.state_dict(), best_val_loss,
                                total_train_loss, dropoutrate, batch_size, numoutputs, learning_rate,
                               summarystring, additionalsummary)
            print(">>> Saving model with best val loss " + str(best_val_loss))

        scheduler.step()
        lastlr = scheduler.get_last_lr()[0]
        print(lastlr)
        writer.add_scalar('lr', lastlr, epoch)

    print("Best val loss: %.6f"%(best_val_loss))
    writer.close()
    writerval.close()

else:
    model = model.eval()

    lossmatrix = []
    with torch.set_grad_enabled(False):
        for batch_idy, (x, y) in enumerate(val_generator):
            x, y = x.to(device), y.to(device)
            y_predicted = model(x)
            cost = lossfunction(y_predicted, y, reduction='none')
            cost = cost.cpu().numpy()
            #print(cost)
            assert cost.shape[1] == numoutputs
            lossmatrix.append(cost)

        lossmatrix = np.concatenate(lossmatrix)

        print("##########################################")
        for outputfeature in range(numoutputs):
            losses = lossmatrix[:, outputfeature]
            mean = np.mean(losses)
            std = np.std(losses)

            print("Output " + str(outputfeature) + " error: mean = " + str(np.round(mean, 4)) + ", std = " + str(np.round(std, 4)))

        print("Error based on normalization range [-1.0, 1.0]! If error output here is e.g. 0.05 --> error is 2.5% !")
        print("Best val loss was " + str(np.round(checkpoint['best_val_loss'], 4))
              + ", train loss was " + str(np.round(checkpoint['total_train_loss'],4)))
        # Mean error: what is the mean L1 loss across all samples in the validation set?
        # Std: how much do the losses deviate from the above mean in the validation set?
        # (Low: The mean error reported above for that feature is roughly that )
        print("##########################################")

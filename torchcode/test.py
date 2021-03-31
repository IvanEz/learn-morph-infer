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

from network import *

def test(num_epochs, model, train_dataset, train_generator, optimizer, device, lossfunction, writer,
              numberofbatches, numberofbatches_val,step,
              reduce_lr_on_flat, scheduler, val_generator, step_val, best_val_loss, savelogdir, writerval,
              dropoutrate, batch_size, numoutputs, learning_rate, lr_scheduler_rate, starttrain, endtrain, startval,
              endval, version, schedulername, lossfunctionname, seed, is_debug, optimizername, weight_decay_sgd,
              modelfun_name, lr_patience, includesft, outputmode, args, summarystring, additionalsummary
              ):

    model = model.eval()

    lossmatrix = []
    ys = []
    yspredicted = []

    lambdas = []
    mus = []
    velocities = []

    sqrtDTs = []

    for path in (train_dataset.all_paths + val_dataset.all_paths):
        with open(path + "parameter_tag2.pkl", "rb") as par:
            params = pickle.load(par)
            Dw = params['Dw']  # cm^2 / d
            rho = params['rho']  # 1 / d
            Tend = params['Tend']  # d

            lambdaw = np.sqrt(Dw / rho)  # cm
            mu = Tend * rho  # constant
            velocity = 2 * np.sqrt(Dw * rho)  # cm / d

            sqrtDT = np.sqrt(Dw * Tend)  # cm

            lambdas.append(lambdaw)
            mus.append(mu)
            velocities.append(velocity)
            sqrtDTs.append(sqrtDT)

    lambda_min = np.min(lambdas)
    lambda_max = np.max(lambdas)
    mu_min = np.min(mus)
    mu_max = np.max(mus)
    velocity_min = np.min(velocities)
    velocity_max = np.max(velocities)

    sqrtmu_min = np.min(np.sqrt(mus))
    sqrtmu_max = np.max(np.sqrt(mus))

    sqrtDT_min = np.min(sqrtDTs)
    sqrtDT_max = np.max(sqrtDTs)

    with torch.set_grad_enabled(False):
        for batch_idy, (x, y) in enumerate(val_generator):
            x, y = x.to(device), y.to(device)
            y_predicted = model(x)
            cost = F.l1_loss(y_predicted, y, reduction='none')
            cost = cost.cpu().numpy()
            # print(cost)
            cost = cost / 2  # takes into account that normalization range is [-1,1]
            # print(cost)
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

            print("Output " + str(outputfeature) + " error: mean = " + str(np.round(mean, 4)) + ", std = " + str(
                np.round(std, 4)))

            plt.figure()
            plt.hist(losses, bins=100, range=(0.0, 1.0))
            plt.title(loaddir + ": \n " + str(outputfeature) + " - outputmode: " + str(outputmode))
            plt.savefig("fig" + str(outputfeature) + ".png")

        # print("Error based on normalization range [-1.0, 1.0]! If error output here is e.g. 0.05 --> error is 2.5% !")
        print("Best val loss was " + str(np.round(checkpoint['best_val_loss'], 6))
              + ", train loss was " + str(np.round(checkpoint['total_train_loss'], 6)))
        # Mean error: what is the mean L1 loss across all samples in the validation set?
        # Std: how much do the losses deviate from the above mean in the validation set?
        # (Low: The mean error reported above for that feature is roughly that )
        print("##########################################")

        np.savez_compressed("results_" + str(outputmode) + ".npz", losses=lossmatrix, ys=ys,
                            yspredicted=yspredicted)  # still in normalization range [-1,1] !

        print("############# RANGED ERROR ##################")
        # Calculates error not based on the theoretical minimum and maximum range, but based on the
        # minimum and maximum observed values in the dataset (since theoretical range much bigger for some vars)
        # through combination
        '''
        lambdas_normalized = ys[:,2] #all lambdaw
        lambdas = np.interp(lambdas_normalized, mycode.normalization_range, [np.sqrt(0.001), np.sqrt(7.5)])

        predicted_lambdas_normalized = yspredicted[:, 2]
        predicted_lambdas = np.interp(predicted_lambdas_normalized, mycode.normalization_range, [np.sqrt(0.001), np.sqrt(7.5)])

        lambdas = np.interp(lambdas, [lambda_min, lambda_max], [0.0, 1.0])
        predicted_lambdas = np.interp(predicted_lambdas, [lambda_min, lambda_max], [0.0, 1.0])

        ranged_error_lambda = np.abs(predicted_lambdas - lambdas)

        mean = np.mean(ranged_error_lambda)
        std = np.std(ranged_error_lambda)

        print("Ranged lambda error: mean = " + str(np.round(mean, 4)) + ", std = " + str(np.round(std, 4)))

        plt.figure()
        plt.hist(ranged_error_lambda, bins=100, range=(0.0, 1.0))
        plt.title(loaddir + ": \n " + "ranged error lambda")
        plt.savefig("fig" + "RANGEDLAMBDAERROR" + ".png")
        '''
        #####################################################################################

        if outputmode == 0:
            ranged_error("lambda", 2, [np.sqrt(0.001), np.sqrt(7.5)], ys, yspredicted, [lambda_min, lambda_max],
                         loaddir)
            ranged_error("mu", 3, [0.1, 300.0], ys, yspredicted, [mu_min, mu_max], loaddir)
            ranged_error("v", 4, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)], ys, yspredicted, [velocity_min, velocity_max],
                         loaddir)
        elif outputmode == 1:
            ranged_error("lambda", 0, [np.sqrt(0.001), np.sqrt(7.5)], ys, yspredicted, [lambda_min, lambda_max],
                         loaddir)
            ranged_error("mu", 1, [0.1, 300.0], ys, yspredicted, [mu_min, mu_max], loaddir)
            ranged_error("v", 2, [2 * np.sqrt(4e-7), 2 * np.sqrt(0.003)], ys, yspredicted, [velocity_min, velocity_max],
                         loaddir)
        elif outputmode == 3:
            ranged_error("lambda", 0, [np.sqrt(0.001), np.sqrt(7.5)], ys, yspredicted, [lambda_min, lambda_max],
                         loaddir)
            ranged_error("mu", 1, [0.1, 300.0], ys, yspredicted, [mu_min, mu_max], loaddir)
        elif outputmode == 4:
            ranged_error("sqrt(DT)", 0, [0.1, np.sqrt(22.5)], ys, yspredicted, [sqrtDT_min, sqrtDT_max], loaddir)
            ranged_error("sqrt(Tp)", 1, np.sqrt([0.1, 300.0]), ys, yspredicted, [sqrtmu_min, sqrtmu_max], loaddir)
        print("############# RANGED ERROR ##################")
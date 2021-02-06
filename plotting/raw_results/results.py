import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline

show_annotation = True

def plot(x,y,label,ax,color='nocolor'):
    #plt.scatter(x, y)
    #cs = UnivariateSpline(x, y)
    #plt.plot(x, y)
    x = np.array(x)
    y = np.array(y)
    #z = np.array(z)
    #plt.errorbar(x, y, z, capsize=3, fmt='o', linestyle='--', label=label)
    if color != 'nocolor':
        plt.plot(x,y, 'o', label=label, linestyle='--', color=color)
    else:
        plt.plot(x,y, 'o', label=label, linestyle='--')
    #ax.fill_between(x, y+z, y-z, alpha=0.1)

#variance: amount of uncertainty of the error around bias

xvalues = np.linspace(5000, 100000, 2)
####################################################################################
resnetinv2deeper = {'x': [6400.0, 16000.0, 32000.0, 32000.0, 64000.0, 64000.0, 80000.0], 'y': [0.1492, 0.13605, 0.1177, 0.12, 0.1152, 0.11235, 0.1104]}
resnetinv2deeper_txt = ["1810-21-09-18", "2510-21-15-33", "1910-07-40-02", "2010-09-12-26", "2210-20-36-13", "2410-08-01-59", "2610-09-53-46"]


plt.scatter(resnetinv2deeper['x'], resnetinv2deeper['y'], label="resnetinv2deeper")
if show_annotation:
    for i, txt in enumerate(resnetinv2deeper_txt):
        plt.annotate(txt, (resnetinv2deeper['x'][i], resnetinv2deeper['y'][i]), size=8)

resnetinv2deeperavgpool = {'x': [32000.0], 'y': [0.11235]}
resnetinv2deeperavgpool_txt = ["3110-16-41-35"]

#plt.scatter(resnetinv2deeperavgpool['x'], resnetinv2deeperavgpool['y'], label="resnetinv2deeperavgpool")
#if show_annotation:
#    for i, txt in enumerate(resnetinv2deeperavgpool_txt):
#        plt.annotate(txt, (resnetinv2deeperavgpool['x'][i], resnetinv2deeperavgpool['y'][i]), size=8)
####################################################################################
convnet = {'x': [16000.0, 32000.0, 64000.0], 'y': [0.137, 0.1259, 0.1177]}
convnet_txt = ["2110-23-55-00", "2010-21-22-53", "2010-21-25-51"]

plt.scatter(convnet['x'], convnet['y'], label="convnet")

if show_annotation:
    for i, txt in enumerate(convnet_txt):
        plt.annotate(txt, (convnet['x'][i], convnet['y'][i]), size=8)


plt.legend()
plt.xscale('log')
plt.xlabel("Number of samples in training set")
plt.ylabel("Error validation set (size = 10% * training set size)")
plt.title("Val performance of different models stopped \n when train loss close to val loss (no / low overfit)")
#plt.show()
plt.savefig("models.png")

#performance slightly worse than reported above due to sometimes slightly wrong (too early / too late) saving time in v1.
########################################################################################
resnetinv2deeper = {'x': [6400.0, 16000.0, 32000.0, 64000.0, 80000.0], 'y': [0.1519, 0.1369, 0.1221, 0.11235, 0.11245],
                    'uth_mean': [0.2502, 0.2451, 0.2436, 0.2415, 0.2411], 'uth_std': [0.1562, 0.148, 0.1456, 0.1485, 0.147],
                    'Dw_mean': [0.2405, 0.2202, 0.1936, 0.1693, 0.1694], 'Dw_std': [0.1929, 0.173, 0.153, 0.1556, 0.1526],
                    'p_mean': [0.2391, 0.2218, 0.1989, 0.1848, 0.1871], 'p_std': [0.1608, 0.1493, 0.1511, 0.1461, 0.1421],
                    'Tend_mean': [0.1641, 0.1583, 0.1462, 0.1298, 0.131], 'Tend_std': [0.1705, 0.1677, 0.1588, 0.1458, 0.1516],
                    'icx_mean': [0.0528, 0.0353, 0.0273, 0.0205, 0.0219], 'icx_std': [0.045, 0.0324, 0.03, 0.0245, 0.026],
                    'icy_mean': [0.0514, 0.0318, 0.0216, 0.0167, 0.0173], 'icy_std': [0.042, 0.0271, 0.0232, 0.02, 0.0196],
                    'icz_mean': [0.0518, 0.0316, 0.026, 0.0211, 0.0197], 'icz_std': [0.041, 0.0283, 0.0255, 0.0204, 0.0201]}
resnetinv2deeper_txt = ["1810-21-09-18", "2510-21-15-33", "1910-07-40-02", "2410-08-01-59", "2610-09-53-46"]


convnet = {'x': [16000.0, 32000.0, 64000.0], 'y': [0.137, 0.1272, 0.1172],
                    'uth_mean': [0.2478, 0.246, 0.2437], 'uth_std': [0.1511, 0.1515, 0.1486],
                    'Dw_mean': [0.2164, 0.2015, 0.18], 'Dw_std': [0.1627, 0.1621, 0.1457],
                    'p_mean': [0.2179, 0.2095, 0.1902], 'p_std': [0.1449, 0.1426, 0.142],
                    'Tend_mean': [0.1583, 0.1561, 0.138], 'Tend_std': [0.1616, 0.1626, 0.1508],
                    'icx_mean': [0.0379, 0.0298, 0.0245], 'icx_std': [0.0341, 0.0298, 0.027],
                    'icy_mean': [0.0321, 0.0219, 0.0181], 'icy_std': [0.0295, 0.0231, 0.0206],
                    'icz_mean': [0.034, 0.0264, 0.0222], 'icz_std': [0.0289, 0.0252, 0.022]}
convnet_txt = ["2110-23-55-00", "2010-21-22-53", "2010-21-25-51"]



plt.figure()
plt.xscale('log')
ax = plt.gca()
plot(resnetinv2deeper['x'], resnetinv2deeper['uth_mean'],'uth', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['Dw_mean'], 'Dw', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['p_mean'], 'p', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['Tend_mean'], 'Tend', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['icx_mean'], 'icx', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['icy_mean'], 'icy', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['icz_mean'], 'icz', ax)
#mean and std evaluated on fixed validation set of 8000 tumors
#these errors may be slighlty different than previous plot for each dimension since these values calculated after training time, model is saved a little before optimal point (optimal point determined by hand)
#plt.scatter(resnetinv2deeper['x'], resnetinv2deeper['uth_mean'], )
#plt.ylim(0.0, 0.5)
plt.yticks(np.arange(0.0, 0.3, 0.02))
plt.xlabel("Number of samples in training set")
plt.ylabel("Error mean on fixed validation set: sorted(glob(80000 - 88000))")
plt.title("Performance ResNetInv2Deeper (Error mean)")
plt.legend(loc=2, bbox_to_anchor=(0.975,1), borderaxespad=0.)
#plt.show()
plt.savefig("errorresnetinv2.png")

plt.figure()
plt.xscale('log')
ax = plt.gca()
plot(resnetinv2deeper['x'], resnetinv2deeper['uth_std'],'uth', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['Dw_std'], 'Dw', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['p_std'], 'p', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['Tend_std'], 'Tend', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['icx_std'], 'icx', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['icy_std'], 'icy', ax)
plot(resnetinv2deeper['x'], resnetinv2deeper['icz_std'], 'icz', ax)
#mean and std evaluated on fixed validation set of 8000 random tumors
#these errors may be slighlty different than previous plot for each dimension since these values calculated after training time, model is saved a little before optimal point (optimal point determined by hand)
#plt.scatter(resnetinv2deeper['x'], resnetinv2deeper['uth_mean'], )
#plt.ylim(0.0, 0.5)
plt.yticks(np.arange(0.0, 0.3, 0.02))
plt.xlabel("Number of samples in training set")
plt.ylabel("Error std on fixed validation set: sorted(glob(80000 - 88000))")
plt.title("Performance ResNetInv2Deeper (Error std)")
plt.legend(loc=0, bbox_to_anchor=(1,1))
#plt.show()
plt.savefig("stdresnetinv2.png")




#Training of unthresholded (GT volume distribution) -> (sqrt(D/p), Tp, x,y,z) are ["2810-12-51-15", "2810-13-35-54", "2910-08-20-22", "2910-15-44-21"], last one has best performance
##################################################################
############ TRAINING WITH 2 THRESHOLDS ##########################

resnetinv2deeper_2thrs_unoptimized = {'x': [64000.0], 'y': [0.1168],
                                'uth1_mean': [0.2205], 'uth1_std': [0.1515],
                                'uth2_mean': [0.1753], 'uth2_std': [0.1345],
                                'Dw_mean': [0.163], 'Dw_std': [0.1457],
                                'p_mean': [0.1864], 'p_std': [0.1475],
                            'Tend_mean:': [0.1304], 'Tend_std': [0.1463],
                              'icx_mean': [0.0213], 'icx_std': [0.0194],
                              'icy_mean': [0.0159], 'icy_std': [0.014],
                              'icz_mean': [0.0208], 'icz_std': [0.0187]
                                }
resnetinv2deeper_2thrs_unoptimized_txt = ["0111-16-11-42"]
#between 64000.0 with 2 thresholds and 1 threshold THERE IS NO QUALITATIVE DIFFERENCE IN THE ERROR DISTRIBUTIONS!
#####################################################################
resnetinv2deeper_2thrs_optimized = {'x': [64000.0, 80000.0], 'y': [0.105, 0.10674],
                                'uth1_mean': [0.2063, 0.2086], 'uth1_std': [0.1406, 0.1406],
                                'uth2_mean': [0.1331, 0.1336], 'uth2_std': [0.1049, 0.1068],
                                'Dw_mean': [0.1499, 0.1502], 'Dw_std': [0.1317, 0.1322],
                                'p_mean': [0.1786, 0.1793], 'p_std': [0.1355, 0.1345],
                            'Tend_mean:': [0.1216, 0.1252], 'Tend_std': [0.1478, 0.1488],
                              'icx_mean': [0.0185, 0.0197], 'icx_std': [0.0171, 0.019],
                              'icy_mean': [0.0143, 0.017], 'icy_std': [0.0138, 0.0156],
                              'icz_mean': [0.0176, 0.0195], 'icz_std': [0.0163, 0.0178]
                                }
resnetinv2deeper_2thrs_optimized_txt = ["0311-09-42-03", "0411-22-45-55"]

#NOT ENOUGH EVIDENCE THAT L2 is better with 2thrs and D,p,T: ["0711-10-43-27", "0611-23-38-08"]
####################################################################
#NUM_THRESHOLDS = 100 or > 1 leads to big oscillations in validation loss!!!
#WITH PET SCAN:
#"0711-12-45-47": first with pet, starts overfitting from the beginning
#"0711-17-53-22": works but has oscillations, using l2+convnet+weightdecay+nodropout leads also to oscillations ("1011-10-16-13"), reducing numthrs=2 ("0911-11-13-26") does also not solve the problem, unsuccessful trainings with l2: "1011-16-15-27", "1011-14-50-20"

#now y is not error but validation loss, if trained with l2:
resnetinv2deeper_pet_new = {'x': [64000.0, 64000.0, 64000.0, 64000.0, 64000.0, 64000.0, 64000.0, 80000.0, 80000.0], 'y_loss': [0.10344, np.sqrt(0.02), np.sqrt(0.035153), np.sqrt(0.027413), np.sqrt(0.026294), np.sqrt(0.022221), np.sqrt(0.020252), np.sqrt(0.019457), np.sqrt(0.017834)],
                            'uth1_mean': [0.0555, 0.0265, 0.0684, 0.0512, 0.0415, 0.0358, 0.029, 0.0278, 0.0216], 'uth1_std': [0.06, 0.0248, 0.0704, 0.051, 0.0437, 0.0354, 0.0272, 0.027, 0.0239],
                            'uth2_mean': [0.1082, 0.0507, 0.1191, 0.0887, 0.083, 0.0612, 0.0533, 0.0504, 0.0388], 'uth2_std': [0.099, 0.0497, 0.102, 0.0815, 0.0758, 0.0601, 0.0499, 0.0495, 0.0406],
                            'lambda_mean': [0.035342, 0.025579, 0.026828, 0.021744, 0.022877, 0.020299, 0.018445, 0.016152, 0.014469], 'lambda_std': [0.046677, 0.025797, 0.028976, 0.02334, 0.023616, 0.022374, 0.018162, 0.016978, 0.016108],#ranged error
                            'mu_mean': [0.024648, 0.01784, 0.017915, 0.013425, 0.018076, 0.011864, 0.011426, 0.010414, 0.009027], 'mu_std': [0.038265, 0.031007, 0.026404, 0.02165, 0.027118, 0.01872, 0.015488, 0.015749, 0.011273], #ranged error
                            'v_mean': [0.146411, 0.139124, 0.142096, 0.13922, 0.14227, 0.138953, 0.139406, 0.138459, 0.138004], 'v_std': [0.121849, 0.110687, 0.111292, 0.109908, 0.111722, 0.109659, 0.109596, 0.110003, 0.109686], #ranged error
                            'icx_mean': [0.0223, 0.0193, 0.0258, 0.0225, 0.0235, 0.0215, 0.0191, 0.0197, 0.0146], 'icx_std': [0.0195, 0.0169, 0.0221, 0.0197, 0.0198, 0.0184, 0.0161, 0.017, 0.0129],
                            'icy_mean': [0.0182, 0.0137, 0.0186, 0.0164, 0.0178, 0.0175, 0.0158, 0.0143, 0.0121], 'icy_std': [0.0154, 0.012, 0.0164, 0.0148, 0.0152, 0.0151, 0.0132, 0.0124, 0.0106],
                            'icz_mean': [0.0212, 0.0169, 0.0257, 0.0213, 0.0234, 0.0202, 0.0188, 0.0174, 0.0142], 'icz_std': [0.0181, 0.0155, 0.0209, 0.0185, 0.0193, 0.0174, 0.0157, 0.0157, 0.0128]
}

resnetinv2deeper_pet_new_txt = ["0711-17-53-22", "1011-09-46-45", "1611-12-33-59", "1811-10-25-07", "2211-16-28-14", "2211-18-50-33", "2411-10-33-06 (epoch25)", "2611-21-06-42 (epoch16)", "0512-12-35-48 (epoch 37)"]

resnetinv2deeper_pet_new_modelname = ["ResNetInv2DeeperPool", "ResNetInv2SmallPool", "ResNetInvPreActDirect_Small", "ResNetInvPreActDirect_Medium", "PreActNetConstant_16_n1", "NetConstant_noBN_16_n1", "NetConstant_noBN_32_n1", "NetConstant_noBN_64_n1", "NetConstant_noBN_64_n4_l4"]
resnetinv2deeper_pet_version = ["v3", "v3", "v4", "v4", "v4-exper-wide", "v4-exper-wide", "v4-exper-wide", "v4-exper-wide", "v5"]


plt.figure()
#plt.xscale('log')
ax = plt.gca()
plot([2,4,6,8,10,12,14,16, 18], resnetinv2deeper_pet_new['lambda_mean'], 'lambda', ax)
plot([2,4,6,8,10,12,14,16, 18], resnetinv2deeper_pet_new['mu_mean'], 'mu2^2', ax)
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['v_mean'], 'Tend', ax)
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['icx_mean'], 'icx', ax, 'k')
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['icy_mean'], 'icy', ax, 'k')
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['icz_mean'], 'icz', ax, 'k')
ic = (np.array(resnetinv2deeper_pet_new['icx_mean']) + np.array(resnetinv2deeper_pet_new['icy_mean']) + np.array(resnetinv2deeper_pet_new['icz_mean']))/3
print(ic)
plot([2,4,6,8,10,12,14,16, 18], ic, 'ic', ax)
plt.legend()
plt.savefig("pet2.png")
plt.figure()
#plt.xscale('log')
ax = plt.gca()
plot([2,4,6,8,10,12,14,16, 18], resnetinv2deeper_pet_new['lambda_std'], 'lambda', ax)
plot([2,4,6,8,10,12,14,16, 18], resnetinv2deeper_pet_new['mu_std'], 'mu2^2', ax)
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['v_mean'], 'Tend', ax)
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['icx_mean'], 'icx', ax, 'k')
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['icy_mean'], 'icy', ax, 'k')
#plot([2,4,6,8,10,12,14,16], resnetinv2deeper_pet_new['icz_mean'], 'icz', ax, 'k')
ic = (np.array(resnetinv2deeper_pet_new['icx_std']) + np.array(resnetinv2deeper_pet_new['icy_std']) + np.array(resnetinv2deeper_pet_new['icz_std']))/3
print(ic)
plot([2,4,6,8,10,12,14,16, 18], ic, 'ic', ax)
plt.legend()
plt.savefig("pet2-std.png")





#the problem for oscillations is not num_thresholds ("1111-10-49-24"), it's not updating learning rate every batch ("1111-13-23-07"), it's not b=0.5 ("1111-14-41-23")
#smaller learning rate on "1011-09-46-45" did not change oscillations at least in first few epochs ("1911-20-39-41")

#---------------------------
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
#v4-exper-wide:
#"2211-16-28-14": PreActNetConstant_16_n1 reaches val loss 0.026 which is not bad BUT OSCILLATES STRONGLY --> remove BN!
#"2211-18-50-33": is the same as above BUT HAS NO BATCHNORM -> (slightly) better performance AND HAS NO SUCH OSCILLATIONS!!!
#THIS IS THE REASON WE REMOVE BATCHNORM!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#Fourier transform (1.12.2020)
#We do not notice any significant improvement using the fourier transform (either as a normal channel or with a separate "network" for itself). However the training runs we tried were not particularly wide or deep, significant improvements may be visible with architectures with more parameters (maybe), since these are all models which are underfitting.

#Training runs: 2211-18-50-33 (16 filters, l5) - smallest number of parameters, 2411-10-33-06 (32 filters - has biggest number of parameters: 300k) has best performance
#2911-11-55-58, 3011-15-00-54, 3011-21-01-38 are all fourier runs which do not perform bad at all, but their performance is not as good as anticipated

#We continue training with 32 filters (only slight improvement with 64 filters in shallow architecture), maybe deep architecture + 64 filters better? 64 filters had even more data but performance boost was small (2411-10-33-06, 2611-21-06-42)

#TWONET also does not seem to be better than simply using fourier transform as additional channel (3011-15-00-54 vs 3011-21-01-38)
#Fourier transform

#Fourier transform and space (2911-11-55-58) vs no fourier transform - only space (0212-14-10-44) --> they perform the same

#BN vs NOBN: BN has still oscillations in val loss (0312-16-31-25, 0412-08-56-35) vs NOBN has only small oscillations in valloss (2211-18-50-33)

#Normtail (front part of network not normalized, back part normalized): breaks into nan at some point (1012-21-36-37)

#(Conv->Relu)*4 vs (Conv->Relu)*2 (without norm): 4x (0512-12-35-48) has similar performance to 2x (2611-21-06-42)


####################################################
### EXPERIMENTS WITH NORM BUT WITHOUT NECROTIC/NORMALIZED PET (2.1.2021)
## Normtail (the front is not normalized because volume mostly empty) but the back is, does not work (1012-21-36-37: NaN, 1112-22-15-08: learning rate too small, slow learning)
## exclusively FT (1212-16-47-01) vs NOFT (1812-12-06-23) with normalization: NOFT performs significantly better
####################################################

####################################################
## EXPERIMENTS WITH NECROTIC / NORMALIZED PET (v7)
## first training, slightly asymmetric architecture: 2212-11-54-40 (best performance: 5e-4 for (infiltration length, Tp) after five day training, some generalization gap)
##very small difference between 2212-11-54-40 (with necrotic+normalized pet) vs 1812-12-06-23 (no necrotic / normalized pet) during first epochs
## ConvNormRelu (2712-11-59-37): performed worse, could be due to wrong learning rate, but this was the heuristically found lr (training with small trainset) --> would need more experimentation, however normway could be better because it allows to norm after maxpool
## first normway training: 2312-18-05-49, we see it performs similarly to the one above so we continue experimentation with this


## Two archs: BIG 128 wide and deep with no weight decay (2912-00-14-44) and some weight decay (2912-11-35-58), we continue with weight decay training, but stop since performance is comparable with 2212-11-54-40 which is only 64 wide, so for the moment we continue with 64 filter wide networks (probably 128 has just much more parameters than needed)

## Training a not so deep architecture with no norm (3112-12-56-20) which is the best performing architecture right now vs a deeper architecture with norm (0201-11-39-27). --> the deeper architecture with norm is not significantly better, it is slightly better in terms of performance reached in number of epochs, but not in terms of performance reached after a certain training time, so we prefer simpler architecture because: 1. it is less deep 2. it does not use norm 3. it is faster to train -> since we generate different tumors each epoch, faster training is preferrable as it allows the network to see more tumors in the same span of training time

#We continue with the 3112-12-56-20 arch and as we notice overfitting towards the end of training, we want to see the effect of weight decay=0.05 and smaller batch size 32, so we start 0501-12-51-17 and see that at the beginning of training, the additional regularization does not hurt training, so we can continue using these hyperparameters for the next trainings.

#There is also another training run 2912-11-35-58, but it is too massive and training takes too long, which is why this arch is discarded.

#Before training main arch 0701-09-37-48, we try out the effect of dropout on the last layer (0601-13-59-02), but the effect makes everything too noisy and training looks already superslow in the first few epochs, so we discard using dropout (at least with dropout probability as used here).





#final training for growth: 1401-21-45-28/epoch50
#final training for xyz: 0701-09-37-48/epoch48

#predicting (D,p,T) in ablation analysis: 2201-20-14-05/epoch34

#ONLY MRI (in presention):
#growth: 2801-12-55-55 (overfits early, so there will be probably another training with more regularization)
#xyz: 0502-13-56-41 (training at the time of writing)



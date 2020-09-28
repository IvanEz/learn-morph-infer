import numpy as np
import random
import pickle5 as pickle
from glob import glob
import os
from timeit import default_timer as timer

beginning = 28103 # inclusive
end = 100001 #exclusive

originalfiles = "/mnt/Drive2/ivan_kevin/samples_extended_copy/Dataset/"
#originalfiles = "/home/kevin/Desktop/thresholding/"

newlocation = "/mnt/Drive2/ivan_kevin/samples_extended_thr2/Dataset/"
#newlocation = "/home/kevin/Desktop/thresholding2/"

#all_paths = sorted(glob("{}*/".format(originalfiles)))[beginning:end]
#num = beginning

for k in range(beginning, end):
    path = originalfiles + str(k) + "/"
    #start = timer()
    thr2 = round(0.35 * random.random() + 0.5, 5)

    with open(path + "parameter_tag.pkl", "rb") as par:
        params = pickle.load(par)
        params['uth2'] = thr2

    with np.load(path + "Data_0001.npz") as data:
        volume = data['data']
        #oldthr = data['thr_data']
        result = (volume >= thr2).astype(float)

    #print("Path is " + str(path[len(originalfiles):-1]) + ", num is " + str(num))

    #if str(path[len(originalfiles):-1]) != str(num):
    #    print("WARNING!")

    newdir = newlocation + str(k) + str("/")
    os.makedirs(os.path.dirname(newdir), exist_ok=True)

    #np.savez_compressed(newdir + "Data_0002.npz", data=volume, thr2_data=result, thr_data=oldthr)
    np.savez_compressed(newdir + "Data_0001_thr2.npz", data=volume, thr2_data=result)

    fh = open(newdir + "parameter_tag2.pkl", "wb")
    pickle.dump(params, fh)
    fh.close()
    #num += 1
    #end = timer()
    #print(end - start)
    print(k)

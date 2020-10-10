import shutil
from glob import glob
import os.path

all_paths = sorted(glob("{}/*/".format("/mnt/Drive2/ivan_kevin/samples_extended_copy/Dataset/")))

tomove = all_paths[85000:]
target = "/mnt/Drive2/ivan_kevin/samples_extended_copy_testset/"

for datapoint in tomove:
    print(shutil.move(datapoint, target))

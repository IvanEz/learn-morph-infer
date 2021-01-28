import numpy as np
import scipy as sp
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
#import scipy.interpolate
from os import listdir
import glob
import os
#import h5py
from multiprocessing import Pool
from functools import partial
import argparse
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import pandas as pd
#from scipy.ndimage import map_coordinates, morphology, generate_binary_structure

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="", type=str)
parser.add_argument('--parapid', default=0, type=int)
args = parser.parse_args()

tumorname = args.name
parapid = args.parapid


vtk_path = 'vtus' + str(parapid) + '/'
#npz_path = 'npzs' + str(parapid) + '/'
npz_path = 'npzresults/'

channel = ['0','1']
#channel = ['0','1','2','3', '4']
#channel = ['0']
dir_list = os.listdir(vtk_path)

def read_grid_vtk(data):
    # Get the coordinates of nodes in the mesh
    nodes_vtk_array= data.GetPoints().GetData()
    vertices = vtk_to_numpy(nodes_vtk_array)
    #The "Velocity" field is the vector in vtk file
    numpy_array = []
    for i in channel:
        vtk_array = data.GetPointData().GetArray('channel'+i)
        numpy_array.append(vtk_to_numpy(vtk_array))

    return vertices, np.array(numpy_array)

def extract_VTK(filename):
    # read poly data
    print(filename)
    reader.SetFileName(filename)
    # reader.ReadAllVectorsOn()
    # reader.ReadAllScalarsOn()
    reader.Update()
    vtk_data = reader.GetOutput()

    vertices, numpy_array = read_grid_vtk(vtk_data)

    print(numpy_array.T.shape)
    numpy_data[x, y, z, :] = numpy_array.T

    path, filename = os.path.split(filename)
    file_name = npz_path+os.path.split(path)[1]+"/"+tumorname+filename
    file_name = file_name.replace(".vtu",".npz")
    try:
        os.makedirs(npz_path+os.path.split(path)[1])
    except:
        pass

    np.savez_compressed(file_name, data=numpy_data)
    print("File saved at ", file_name)

first_file = True
for dir in dir_list:
    files_cfd = []
    for filename in sorted(glob.glob(os.path.join(vtk_path+dir,'*.vtu'))):
        print(filename)
        files_cfd.append(filename)

    #if first_file:
    print("One Time computation for First File")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(files_cfd[1]) #files_cfd[2]
    #reader.ReadAllVectorsOn()
    #reader.ReadAllScalarsOn()
    reader.Update()
    vtk_data = reader.GetOutput()

    vertices, numpy_array = read_grid_vtk(vtk_data)
    bounds_cfd = vtk_data.GetBounds()

    print(bounds_cfd, vertices.shape, numpy_array.shape)

    H = np.unique(vertices[:,0]).shape[0]
    W = np.unique(vertices[:,1]).shape[0]
    D = np.unique(vertices[:,2]).shape[0]
    factor = 128
    print(f"{H}, {W}, {D}")

    numpy_data = np.zeros((H, W, D, len(channel)))

    x, y, z = zip(*list(map(tuple, np.uint16(factor*vertices))))

    #first_file = False
    

    pool = Pool(1)
    pool.map(extract_VTK, files_cfd)
    pool.close()

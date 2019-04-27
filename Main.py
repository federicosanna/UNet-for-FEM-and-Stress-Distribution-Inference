import sys
import json
import os
import glob
import time
import tensorflow as tf
import numpy as np
import cv2
import random
import decimal as Decimal
import matplotlib.pyplot as plt
import matplotlib
import torch

import Models
import Generate_Poly as GP
import Train_Denoiser
# import utils

from torch.utils.data import Dataset
from numpy import array, newaxis, expand_dims
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Cropping2D
from keras.layers import Input, UpSampling2D, concatenate

# Generate random seed to allow reproducibility
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

# # Print the shape of the model
# modellino = get_denoise_model((64, 64, 1))
# print(modellino.summary())

n_gons = [4, 5, 6, 7, 8]            # Types of polygons to be contained in the dataset
ds_poly = GP.SlighlyMoreClevr(n_gons=n_gons, canvas_size=64, size_of_ds_poly=6000)  # Generate dataset

# Generate random train and test dataset
random_indices_poly = torch.randperm(len(ds_poly))
train_split_poly = int(0.75 * len(ds_poly))
train_random_indices_poly = random_indices_poly[:train_split_poly]
test_random_indices_poly = random_indices_poly[train_split_poly:]

# Class for loading Polygons sequence from a sequence folder
denoise_generator_poly = GP.DenoiseHPatchesPoly(random_indices_poly=train_random_indices_poly, ds_poly=ds_poly, batch_size=50)
denoise_generator_val_poly = GP.DenoiseHPatchesPoly(random_indices_poly=test_random_indices_poly, ds_poly=ds_poly, batch_size=50)

# Show sample input of the network
# plt.imshow(denoise_generator_poly[9][0][49,:,:,0])
# # plt.show()
# plt.imshow(denoise_generator_poly[9][1][49,:,:,0])
# # plt.show()

shape = (64, 64, 1)
denoise_model = Models.get_denoise_model(shape)
epochs = 1

Train_Denoiser.train_denoiser(denoise_generator_poly, denoise_generator_val_poly, denoise_model, epochs)


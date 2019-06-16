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

# Generate random seed to allow reproducibility
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

# # Print the shape of the model
# modellino = get_denoise_model((64, 64, 1))
# print(modellino.summary())

#   ===================================== Generate Dataset =====================================
#
# # n_gons = [4, 5, 6, 7, 8]            # Types of polygons to be contained in the dataset
# n_gons = [5]            # Types of polygons to be contained in the dataset
# ds_poly = GP.SlighlyMoreClevr(n_gons=n_gons, canvas_size=64, size_of_ds_poly=6000)  # Generate dataset
#
# # Generate random train and test dataset
# split = 0.75
# random_indices_poly = torch.randperm(len(ds_poly))
# train_split_poly = int(split * len(ds_poly))
# train_random_indices_poly = random_indices_poly[:train_split_poly]
# test_random_indices_poly = random_indices_poly[train_split_poly:]
#
# # Class for loading Polygons sequence from a sequence folder
# denoise_generator_poly = GP.DenoiseHPatchesPoly_Exp4(random_indices_poly=train_random_indices_poly, ds_poly=ds_poly, batch_size=50)
# denoise_generator_val_poly = GP.DenoiseHPatchesPoly_Exp4(random_indices_poly=test_random_indices_poly, ds_poly=ds_poly, batch_size=50)
#
# # Show sample input of the network
# # plt.imshow(denoise_generator_poly[9][0][49,:,:,0])
# # # plt.show()
# # plt.imshow(denoise_generator_poly[9][1][49,:,:,0])
# # # plt.show()
#
# shape = (5, 2, 1)
# denoise_model = Models.get_denoise_model_5x2(shape)

#   ===================================== Stage 0 for Shape Reconstuction =====================================

# Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Inputs_DataSet.npy')
#
# # Generate random train and test dataset
# split = 0.8
# random_indices_poly = torch.randperm(len(Inputs))
# train_split_poly = int(split * len(Inputs))
# train_random_indices_poly = random_indices_poly[:train_split_poly]
# test_random_indices_poly = random_indices_poly[train_split_poly:]
#
# # Class for loading Polygons sequence from a sequence folder
# denoise_generator_poly = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=train_random_indices_poly, ds_poly=Inputs, batch_size=50)
# denoise_generator_val_poly = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=test_random_indices_poly, ds_poly=Inputs, batch_size=50)
#
# shape = (64, 64, 1)
# denoise_model = Models.get_denoise_model(shape)

#   ===================================== Stage 0 for Stress Reconstuction =====================================

Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')

# Generate random train and test dataset
split = 0.8
random_indices_poly = torch.randperm(len(Inputs))
train_split_poly = int(split * len(Inputs))
train_random_indices_poly = random_indices_poly[:train_split_poly]
test_random_indices_poly = random_indices_poly[train_split_poly:]

# Class for loading Polygons sequence from a sequence folder
denoise_generator_poly = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=train_random_indices_poly, ds_poly=Inputs, batch_size=50)
denoise_generator_val_poly = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=test_random_indices_poly, ds_poly=Inputs, batch_size=50)

shape = (64, 64, 1)
denoise_model = Models.get_denoise_model(shape)

#   ===================================== Stage 1_3 =====================================

# Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
# Labels = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Inputs_DataSet.npy')
#
# # Generate random train and test dataset
# split = 0.75
# random_indices_poly = torch.randperm(len(Inputs))
# train_split_poly = int(split * len(Inputs))
# train_random_indices_poly = random_indices_poly[:train_split_poly]
# test_random_indices_poly = random_indices_poly[train_split_poly:]
#
# # Class for loading Polygons sequence from a sequence folder
# denoise_generator_poly = GP.DenoiseHPatchesPoly_Stage_1_3(random_indices_poly=train_random_indices_poly, inputs=Inputs, labels=Labels, batch_size=50)
# denoise_generator_val_poly = GP.DenoiseHPatchesPoly_Stage_1_3(random_indices_poly=test_random_indices_poly, inputs=Inputs, labels=Labels, batch_size=50)
#
# shape = (5, 6, 1)
# denoise_model = Models.get_denoise_model_5x6(shape)

#   ===================================== Experiment 5 =====================================

# Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
# Labels = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
#
# # Generate random train and test dataset
# split = 0.75
# random_indices_poly = torch.randperm(len(Inputs))
# train_split_poly = int(split * len(Inputs))
# train_random_indices_poly = random_indices_poly[:train_split_poly]
# test_random_indices_poly = random_indices_poly[train_split_poly:]
#
# # Class for loading Polygons sequence from a sequence folder
# denoise_generator_poly = GP.DenoiseHPatchesPoly_Exp5(random_indices_poly=train_random_indices_poly, inputs=Inputs, labels=Labels, batch_size=50)
# denoise_generator_val_poly = GP.DenoiseHPatchesPoly_Exp5(random_indices_poly=test_random_indices_poly, inputs=Inputs, labels=Labels, batch_size=50)
#
# shape = (64, 64, 4)
# denoise_model = Models.get_denoise_model(shape)

#   ===================================== Train =====================================


epochs = 1

Train_Denoiser.train_denoiser(denoise_generator_poly, denoise_generator_val_poly, denoise_model, epochs)

#   ===================================== Output results =====================================

imgs, imgs_clean = next(iter(denoise_generator_poly))
index = np.random.randint(0, imgs.shape[0])
imgs_den = denoise_model.predict(imgs)

# # Plot difference image
# plt.imshow(imgs_den[index,:,:,0]-imgs_clean[index,:,:,0], cmap='gray')
# plt.show()

image_difference = imgs_den[index,:,:,0]-imgs_clean[index,:,:,0]
image_difference_vector = np.reshape(image_difference, (1,-1))
ID = np.sum(image_difference_vector)
print("ID is:\n"+str(ID/(64*64)))
MSE = np.sum(np.square(image_difference_vector))/(64*64)
print("MSE is:\n"+str(MSE))

GT_vector = np.reshape(imgs_clean[index,:,:,0], (1,-1))
MPE = np.sum(np.multiply(GT_vector,image_difference_vector))
print("MPE is:\n" + str(MPE))

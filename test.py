import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import time
import datetime
from math import sqrt
import Generate_Poly as GP
import matplotlib

gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())

import matplotlib.pyplot as plt
import matplotlib
import torch


from torch.utils.data import Dataset
from numpy import array, newaxis, expand_dims
from keras.models import load_model


import warnings


#   ===================================== Mio Dataset =====================================

# ds_poly = GP.SlighlyMoreClevr(n_gons=[5], canvas_size=64, size_of_ds_poly=6000)  # Generate dataset
# print(type(ds_poly[1]))

#   ===================================== Stage 0 =====================================

# Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Inputs_DataSet.npy')
# # Inputs = torch.from_numpy(Inputs)
# # print(Inputs.shape[0])
# print(Inputs.shape)
# print(Inputs[1][0,0,:,:])     # use for 64x64 images
# # print(Inputs[1][:,:])           # use for 5x6 array
# plt.imshow(Inputs[71][0,0,:,:]) # use for 64x64 images
# # plt.imshow(Inputs[71][:,:])       # use for 5x6 array
# # plt.show()
#
# Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
#
# # Generate random train and test dataset
# split = 0.75
# random_indices_poly = torch.randperm(len(Inputs))
# train_split_poly = int(split * len(Inputs))
# train_random_indices_poly = random_indices_poly[:train_split_poly]
# test_random_indices_poly = random_indices_poly[train_split_poly:]
#
# # Class for loading Polygons sequence from a sequence folder
# # denoise_generator_poly = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=train_random_indices_poly, ds_poly=Inputs, batch_size=50)
# # denoise_generator_val_poly = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=test_random_indices_poly, ds_poly=Inputs, batch_size=50)
#
# index = 1
# dim_in = (5, 6)
# dim_out = (64, 64)
# img_clean = np.empty((50,) + dim_in + (1,))
# img_noise = np.empty((50,) + dim_in + (1,))
# for i in range(50):
#     img_clean[i] = array(Inputs[train_random_indices_poly[index * 50 + i]])[:, :, newaxis]
#     img_noise[i] = array(Inputs[train_random_indices_poly[index * 50 + i]])[:, :, newaxis]
# # print(Inputs[train_random_indices_poly[index * 50 + 5]][0][0].shape)
# print(img_clean.shape)
#   ===================================== Stage 3 =====================================

# Labels = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
# print(Labels[1].shape)
# print(Labels[1][0,0,:,:])     # use for 64x64 images
# plt.imshow(Labels[73][0,0,:,:]) # use for 64x64 images
# plt.show()

# ====================== Check the output of the model ======================

# denoise_model = load_model('./Saved_Models/modellino.model')

# ====================== Input & Output are from same dataset =========================
# Inputs = np.load(
#     '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
# random_indices_poly = torch.randperm(len(Inputs))
# generator = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=random_indices_poly, ds_poly=Inputs, batch_size=50)

# ====================== Input & Output are from different dataset =========================
Inputs = np.load(
    '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
Labels = np.load(
    '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
random_indices_poly = torch.randperm(len(Inputs))
generator = GP.DenoiseHPatchesPoly_Exp5(random_indices_poly=random_indices_poly, inputs=Inputs, labels=Labels,
                                             batch_size=50)

# imgs, imgs_clean = next(iter(generator))
# # index = np.random.randint(0, imgs.shape[0])
# index = 7
# imgs_den = denoise_model.predict(imgs)
# # plt.subplot(131)
# # plt.imshow(imgs[index, :, :, 0], cmap='gray')
# # plt.title('Noisy', fontsize=20)
# # plt.gca().set_xticks([])
# # plt.gca().set_yticks([])
# # plt.subplot(132)
# # plt.imshow(imgs_den[index, :, :, 0], cmap='gray')
# # plt.title('Denoised', fontsize=20)
#
# print(imgs_den[index, 30, :, 0])


# # Generate random train and test dataset
# split = 0.75
# random_indices_poly = torch.randperm(len(Inputs))
# train_split_poly = int(split * len(Inputs))
# train_random_indices_poly = random_indices_poly[:train_split_poly]
# test_random_indices_poly = random_indices_poly[train_split_poly:]
# index = 1
# dim_in = (5, 6)
# dim_out = (64, 64)
# img_clean = np.empty((50,) + dim_in + (1,))
# img_noise = np.empty((50,) + dim_in + (1,))
# for i in range(50):
#     # img_clean[i, :, :, 0] = array(self.labels[self.random_indices_poly[index * self.batch_size + i]][0][0])
#     img_noise[i, :, :, 0] = array(Inputs[train_random_indices_poly[index * 50 + i][[1, 1][1, 2]]])

print(np.asarray(generator[60][0][1][:,:,0]))
plt.imshow(np.asarray(generator[60][0][45][:,:,1])) # use for 64x64 images
plt.show()
# print(Inputs[1000][:, 0])

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


import warnings

# # Inputs = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
# print(Inputs[1].shape)
# # print(Inputs[1][0,0,:,:])     # use for 64x64 images
# print(Inputs[1][:,:])           # use for 5x6 array
# # plt.imshow(Inputs[71][0,0,:,:]) # use for 64x64 images
# plt.imshow(Inputs[71][:,:])       # use for 5x6 array
# plt.show()

Labels = np.load('/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
print(Labels[1].shape)
print(Labels[1][0,0,:,:])     # use for 64x64 images
plt.imshow(Labels[73][0,0,:,:]) # use for 64x64 images
plt.show()
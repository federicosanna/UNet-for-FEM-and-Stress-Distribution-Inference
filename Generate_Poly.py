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

def generate_polygons(rmin, rmax, n_gons, n_out):
    """    Generates n_out polygons of n_gons sides
            from a circle with a radius between rmin and rmax """
    # Creating a list of n_out random radii with values
    # between rmin and rmax (can be redundant)
    listr = np.random.ranf(n_out) * (rmax - rmin) + rmin

    # Initializing the Matrix of angles of size (n_gons, n_out)
    mat_theta = np.zeros((n_gons, n_out))
    thetanormal = [k * 2 * np.pi / n_gons for k in range(n_gons)]

    for i in range(n_out):
        mat_theta[:, i] = [np.random.normal(thetanormal[k], listr[i] / 9) for k in range(n_gons)]

    x = listr * np.cos(mat_theta)  # Xcoordinates
    y = listr * np.sin(mat_theta)  # Ycoordinates

    return (x, y)


def generate_poly_with_variable_n_gons(n_gons, size_of_ds_poly):
    """ Generate a tuple of size size_of_ds_poly, cointaing pairs of arrays
        representing the x and y coordinates. Generates the vertices of polygons
        with n_gons corners.
        n_gons is required to be a list of integers with the numbers of corners
        of the polygons wanted to be part of the dataset.
        The dataset will be split equaly between the different types of polygons.

        Inputs:
        - n_gons (type=list): represent types of polygons that you want to
          be part of the dataset
        - size_of_ds_poly: number of polygons to be included in the dataset
        Output:
        - tuple of size 'size_of_ds_poly' cointaing elements made of two arrays
          each of size (1 x n_gons) representing the coordinates of the vertices.
        Example:
        generate_poly_with_variable_n_gons([5, 6], 3)
            a Tuple with 3 elements corresponding to:
            1 pair of arrays with the pentagon coordinates:
            [x_1,x_2,x_3,x_4,x_5]
            [y_1,y_2,y_3,y_4,y_5]

            2 pairs of arrays with the exagons coordinates:
            [x_11,x_12,x_13,x_14,x_15,x_16]
            [y_11,y_12,y_13,y_14,y_15,y_16]

            [x_21,x_22,x_23,x_24,x_25,x_26]
            [y_21,y_22,y_23,y_24,y_25,y_26]
        """
    # How many images with a certain number of corners
    images_per_n_gons = int(size_of_ds_poly / len(n_gons))
    # Initialise list for the vertices
    vertices = [None] * size_of_ds_poly
    # Since size_of_ds_poly/len(n_gons) could be a non-integer, we need to take
    # care of the last cases with 2 separate for loops
    # Fill the list with poly of different numbers of corners
    if len(n_gons) > 1:
        for i in range(len(n_gons) - 1):
            for j in range(images_per_n_gons):
                vertices[j + i * images_per_n_gons] = generate_polygons(0.9, 0.6, n_gons[i], 1)
    # Last one fills up until the end of the ds size
    for i in range(size_of_ds_poly - (images_per_n_gons * (len(n_gons) - 1))):
        vertices[(images_per_n_gons * (len(n_gons) - 1)) + i] = generate_polygons(0.9, 0.6, n_gons[len(n_gons) - 1], 1)

    return vertices


def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])

    return idxs[1] * sign <= max_col_idx * sign


def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    # Normalising the vertices from the range (-1:1) to (0:63)
    vertices = ((vertices + 1) * (64 / 2)).round()
    # Inverting the order of the vertices to conform to the standatd of the function
    vertices = vertices[::-1]

    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array


# Generate the dataset of the vertices and the polygons


class SlighlyMoreClevr(Dataset):

    def __init__(self, n_gons, canvas_size=64, size_of_ds_poly=6000):
        self.canvas_size = canvas_size
        # vertices is a list of 6000 pairs of arrays [x,y] each with n_corners elements
        #         self.vertices = [
        #             generate_polygons(0.9,0.6,n_gons,1) for i in range(size_of_ds_poly)
        #         ]
        self.vertices = generate_poly_with_variable_n_gons(n_gons, size_of_ds_poly)

    #         self.vertices = [
    #             spatialise_vertices(n_gons) for i in range(size_of_ds_poly)
    #         ]

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        v = self.vertices[index]
        p = create_polygon([64, 64], np.concatenate(v, axis=1))
        return v, p





class DenoiseHPatchesPoly(keras.utils.Sequence):
    """Class for loading a Polygons' sequence from a sequence folder.
        Class to generate the Dataset for the SECOND experiment
        The second experiment consists in making a network reconstruct the image of a
        polygon based on the picture of its corners.
        The function allows to construct a Dataset that contains as Noisy image an
        image of the pixels representing the corners"""

    def __init__(self, ds_poly, random_indices_poly, batch_size):
        # self.all_paths = [] not needed for poly
        self.batch_size = batch_size
        self.random_indices_poly = random_indices_poly
        self.dim = (64, 64)
        self.n_channels = 1
        self.on_epoch_end()
        self.ds_poly = ds_poly

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.random_indices_poly) / self.batch_size))

    def obtain_corners(self, v):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v[0] + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v[1] + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.ds_poly.canvas_size, self.ds_poly.canvas_size, 1), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1][0] = 1
        return base_array.numpy()

    def __getitem__(self, index):
        # be picked up
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            img_clean[i] = array(self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][1])[:, :, newaxis]
            img_noise[i] = self.obtain_corners(self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][0])

        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.random_indices_poly = self.random_indices_poly[torch.randperm(len(self.random_indices_poly))]


class DenoiseHPatchesPoly_Stage_0(keras.utils.Sequence):
    """Class for loading a Polygons' sequence from a Marcin's Data folder.
        Class to generate the Dataset for the experiment of Stage 0 (from shape to shape)
        """

    def __init__(self, ds_poly, random_indices_poly, batch_size):
        # self.all_paths = [] not needed for poly
        self.batch_size = batch_size
        self.random_indices_poly = random_indices_poly
        self.dim = (64, 64)
        self.n_channels = 1
        self.on_epoch_end()
        self.ds_poly = ds_poly

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.random_indices_poly) / self.batch_size))

    def __getitem__(self, index):
        # be picked up
        # shape of img_noise is (batch_size, 64, 64, 1)
        # shape of img_noise is (batch_size, 64, 64, 1)
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            img_clean[i] = array(self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][0][0])[:, :, newaxis]
            img_noise[i] = array(self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][0][0])[:, :, newaxis]

        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.random_indices_poly = self.random_indices_poly[torch.randperm(len(self.random_indices_poly))]

class DenoiseHPatchesPoly_Stage_1_3(keras.utils.Sequence):
    """Class for loading a 5x6 matrix of parameters and a 64x64 image with the Von Misses stresses
        from a Marcin's Data folder.
        Class to generate the Dataset for the experiment of Stage 1_3 (from 5x6 parameters to Stresses)
        """

    def __init__(self, inputs, labels, random_indices_poly, batch_size):
        # self.all_paths = [] not needed for poly
        self.batch_size = batch_size
        self.random_indices_poly = random_indices_poly
        self.dim_out = (64, 64)
        self.dim_in = (5, 6)
        self.n_channels = 1
        self.on_epoch_end()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.random_indices_poly) / self.batch_size))

    def __getitem__(self, index):
        # be picked up
        # shape of img_noise is (batch_size, 5, 6, 1)
        # shape of img_noise is (batch_size, 64, 64, 1)
        img_clean = np.empty((self.batch_size,) + self.dim_out + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim_in + (self.n_channels,))
        for i in range(self.batch_size):
            img_clean[i] = array(self.labels[self.random_indices_poly[index * self.batch_size + i]][0][0])[:, :, newaxis]
            img_noise[i] = array(self.inputs[self.random_indices_poly[index * self.batch_size + i]])[:, :, newaxis]

        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.random_indices_poly = self.random_indices_poly[torch.randperm(len(self.random_indices_poly))]


# Function to generate the Dataset for the FOURTH experiment
# The fourth experiment consists in making a network reconstruct the image of a
# pentagon based on the coordinates of its corners.
# The function allows to construct a Dataset that contains as Noisy image an
# array 5x2 with the coordinates of the corners

from numpy import array, newaxis, expand_dims


class DenoiseHPatchesPoly_Exp4(keras.utils.Sequence):
    """Class for loading an Polygons sequence from a sequence folder"""

    def __init__(self, ds_poly, random_indices_poly, batch_size):
        # self.all_paths = [] not needed for poly
        self.ds_poly = ds_poly
        self.batch_size = batch_size
        self.random_indices_poly = random_indices_poly
        self.dim_In = (5, 2)
        self.dim_Out = (64, 64)
        self.n_channels = 1
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.random_indices_poly) / self.batch_size))

    def __getitem__(self, index):
        # For now, image noisy and image clean are the same thing
        # be picked up
        img_clean = np.empty((self.batch_size,) + self.dim_Out + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim_In + (self.n_channels,))
        for i in range(self.batch_size):
            img_clean[i] = array(self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][1])[:, :, newaxis]
            # img_noise[i] = array(self.ds_poly[self.random_indices_poly[index*self.batch_size+i]][1])[:, :, newaxis]
            img_noise[i] = np.concatenate((self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][0][0],
                                           self.ds_poly[self.random_indices_poly[index * self.batch_size + i]][0][1]),
                                          axis=1)[:, :, newaxis]

        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.random_indices_poly = self.random_indices_poly[torch.randperm(len(self.random_indices_poly))]
#         random.shuffle(self.all_paths)



# Function to generate the Dataset for the Fifth experiment
# The fifth experiment consists in making a network reconstruct the image of the
# stresses on the polygon.
# Images would be having 4 input channels (64,64,4):
#    - one being an array only of zeros and ones, where the ones represent the corners of the polygon;
#    - a scalar field indicating where and what forces  in the X direction are applied (which following the
#       example of Marcin and Kamil would be non-zero only in correspondence of the corners)
#    - a scalar field indicating where and what forces  in the Y direction are applied
#    - a scalar field containing the BC information

class DenoiseHPatchesPoly_Exp5(keras.utils.Sequence):
    """Class for loading an Polygons sequence from a sequence folder
        The fifth experiment consists in making a network reconstruct the image of the
        stresses on the polygon.
     Images would be having 4 input channels (64,64,4):
        - one being an array only of zeros and ones, where the ones represent the corners of the polygon;
        - a scalar field indicating where and what forces  in the X direction are applied (which following the example of Marcin and Kamil would be non-zero only in correspondence of the corners)
        - a scalar field indicating where and what forces  in the Y direction are applied
        - a scalar field containing the BC information
     Output should be the Von Misses stress on the polygon
    """

    def __init__(self, inputs, labels, random_indices_poly, batch_size):
        # self.all_paths = [] not needed for poly
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.random_indices_poly = random_indices_poly
        self.dim = (64, 64)
        self.in_channels = 4
        self.out_channels = 1
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.random_indices_poly) / self.batch_size))

    def obtain_corners(self, v_x, v_y):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v_x + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v_y + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.dim[0], self.dim[1]), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1] = 1
        return base_array.numpy()

    def forces_at_corners(self, v_x, v_y, forces_to_apply):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v_x + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v_y + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.dim[0], self.dim[1]), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1] = forces_to_apply[k]
        return base_array.numpy()

    def fix_corners(self, v_x, v_y, corners_to_fix):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v_x + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v_y + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.dim[0], self.dim[1]), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            if corners_to_fix[k] == 1:
                base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1] = 1
        return base_array.numpy()

    def __getitem__(self, index):
        # For now, image noisy and image clean are the same thing
        # be picked up
        img_clean = np.empty((self.batch_size,) + self.dim + (self.out_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.in_channels,))
        for i in range(self.batch_size):
            img_clean[i, :, :, 0] = array(self.labels[self.random_indices_poly[index * self.batch_size + i]][0][0])
            # In the first input channel place an array only of zeros and ones, where the ones represent the corners of
            # the polygon. To do so, pass the coordinates of the vertices to the function: obtain_corners.
            img_noise[i, :, :, 0] = self.obtain_corners(self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                                                        self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1])
            # In the second input channel place an a scalar field indicating where and what forces  in the X direction
            # are applied (which following the example of Marcin and Kamil would be non-zero only in correspondence of
            # the corners)
            img_noise[i, :, :, 1] = self.forces_at_corners(
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 4])
            # In the third input channel place an a a scalar field indicating where and what forces  in the Y direction
            # are applied
            img_noise[i, :, :, 2] = self.forces_at_corners(
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 5])
            # In the fourth input channel place a scalar field containing the BC information
            img_noise[i, :, :, 3] = self.fix_corners(
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 2])
        return img_noise, img_clean

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.random_indices_poly = self.random_indices_poly[torch.randperm(len(self.random_indices_poly))]
#         random.shuffle(self.all_paths)

class DenoiseHPatchesPoly_Exp6(keras.utils.Sequence):
    """Class for loading an Polygons sequence from a sequence folder
        The Sixth experiment consists in making a network reconstruct the image of the
        stresses on the polygon, and includes a normalisation function. This experiment differs
        from the fifth only for the normalisation.
     Images would be having 4 input channels (64,64,4):
        - one being an array only of zeros and ones, where the ones represent the corners of the polygon;
        - a scalar field indicating where and what forces  in the X direction are applied (which following the example of Marcin and Kamil would be non-zero only in correspondence of the corners)
        - a scalar field indicating where and what forces  in the Y direction are applied
        - a scalar field containing the BC information
     Output should be the Von Misses stress on the polygon
    """

    def __init__(self, inputs, labels, random_indices_poly, batch_size):
        # self.all_paths = [] not needed for poly
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.random_indices_poly = random_indices_poly
        self.dim = (64, 64)
        self.in_channels = 4
        self.out_channels = 1
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.random_indices_poly) / self.batch_size))

    def obtain_corners(self, v_x, v_y):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v_x + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v_y + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.dim[0], self.dim[1]), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1] = 1
        return base_array.numpy()

    def forces_at_corners(self, v_x, v_y, forces_to_apply):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v_x + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v_y + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.dim[0], self.dim[1]), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1] = forces_to_apply[k]
        return base_array.numpy()

    def fix_corners(self, v_x, v_y, corners_to_fix):
        # From vertices in the range -1,1 to vertices in the range 1,64
        vertices_x = ((v_x + 1) * (64 / 2)).round()  # need to be carefull that they end up arriving to 64, not 63
        vertices_y = ((v_y + 1) * (64 / 2)).round()
        # Create 64x64 zero tensor
        base_array = torch.zeros((self.dim[0], self.dim[1]), dtype=torch.uint8)
        # Populate it with ones in correspondance of vertices
        for k in range(vertices_x.shape[0]):
            if corners_to_fix[k] == 1:
                base_array[int(vertices_x[k]) - 1][int(vertices_y[k]) - 1] = 1
        return base_array.numpy()

    def __getitem__(self, index):
        # For now, image noisy and image clean are the same thing
        # be picked up
        img_clean = np.empty((self.batch_size,) + self.dim + (self.out_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.in_channels,))
        G = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            img_clean[i, :, :, 0] = array(self.labels[self.random_indices_poly[index * self.batch_size + i]][0][0])
            img_clean[i, :, :, 0] = img_clean[i, :, :, 0]
            # In the first input channel place an array only of zeros and ones, where the ones represent the corners of
            # the polygon. To do so, pass the coordinates of the vertices to the function: obtain_corners.
            img_noise[i, :, :, 0] = self.obtain_corners(self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                                                        self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1])
            # In the second input channel place an a scalar field indicating where and what forces  in the X direction
            # are applied (which following the example of Marcin and Kamil would be non-zero only in correspondence of
            # the corners)
            img_noise[i, :, :, 1] = self.forces_at_corners(
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 4])
            # In the third input channel place an a a scalar field indicating where and what forces  in the Y direction
            # are applied
            img_noise[i, :, :, 2] = self.forces_at_corners(
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 5])

            # In the fourth input channel place a scalar field containing the BC information
            img_noise[i, :, :, 3] = self.fix_corners(
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 0],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 1],
                self.inputs[self.random_indices_poly[index * self.batch_size + i]][:, 2])
            eps = 0.00001

            Gx = np.max(img_noise[i, :, :, 1]) +eps
            Gy = np.max(img_noise[i, :, :, 2]) +eps
            G[i] = (np.sqrt(Gx ** 2 + Gy ** 2))
            img_noise[i, :, :, 1] = img_noise[i, :, :, 1] / np.abs(Gx)
            img_noise[i, :, :, 2] = img_noise[i, :, :, 2]/ np.abs(Gy)
            img_clean[i, :, :, 0] = img_clean[i, :, :, 0]/ G[i]

        return [img_noise, img_clean]

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.random_indices_poly = self.random_indices_poly[torch.randperm(len(self.random_indices_poly))]

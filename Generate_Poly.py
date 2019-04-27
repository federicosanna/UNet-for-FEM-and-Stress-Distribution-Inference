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
        >>>>a Tuple with 3 elements corresponding to:
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

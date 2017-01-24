from __future__ import print_function

from scipy.misc import imresize
from scipy.misc import imread
from scipy.misc import toimage

from multiprocessing import Pool

import itertools
import cv2 
import h5py
import numpy as np
import os
import random
import sys

from random import shuffle

from keras import backend as K
import theano.tensor as T
import theano
import keras.models
import keras.callbacks

from theano.compile.sharedvalue import shared
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Input, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

img_rows = 512
img_cols = 512
nb_epoch = 1000
iteration_size = 100000
mini_batch_size = 12
delta = 16
initial_discount = 0.01
discount_step = 0.1

num_samples_per_epoch = 50000
num_validation_samples = 5000

d = shared(initial_discount, name = 'd')

def fcrn_loss(y_true, y_pred):
  loss = K.square(y_pred - y_true)
  
  images = []
  
  for i in range(0, mini_batch_size):
    c = y_true[i, 6, :,:].reshape((1, delta, delta))   # The last feature map in the true vals is the 'c' matrix
    
    final_c = (c * loss[i,6,:,:])
    
    c = T.set_subtensor(c[(c<=0.0).nonzero()], d.get_value())
    
    # Element-wise multiply of the c feature map against all feature maps in the loss
    final_loss_parts = [(c * loss[i, j, :, :].reshape((1, delta, delta))).reshape((1, delta, delta)) for j in range(0, 6)]
    final_loss_parts.append(final_c)
    
    images.append(K.concatenate(final_loss_parts))
    
  return K.mean(K.concatenate(images).reshape((mini_batch_size, 7, delta, delta)), axis = 1)
  
class DiscountCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    print("Running callback: " + str(epoch))
    d.set_value(d.get_value() + discount_step)
    
def build_model():
  if os.path.exists(model_file + ".h5"):
    print("Loading saved model for incremental training...")
    model = keras.models.load_model(model_file + ".h5", custom_objects = {'fcrn_loss', fcrn_loss})
  else:
    model = Sequential()
    
    # Layer 1
    model.add(ZeroPadding2D(padding = (2, 2), input_shape=(1, img_rows, img_cols)))
    model.add(Convolution2D(64, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 1: " + str(model.layers[-1].output_shape))
    
    # Layer 2
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 2: " + str(model.layers[-1].output_shape))
    
    # Layer 3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 3: " + str(model.layers[-1].output_shape))
 
    # Layer 4
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 4: " + str(model.layers[-1].output_shape))
    
    # Layer 5
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 5: " + str(model.layers[-1].output_shape))
    
    
    # Layer 6
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 6: " + str(model.layers[-1].output_shape))
    
    
    # Layer 7
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 7: " + str(model.layers[-1].output_shape))
    
    # Layer 8
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 4: " + str(model.layers[-1].output_shape))
    
    
    # Layer 9
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 9: " + str(model.layers[-1].output_shape))
    

    # Layer 10
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(7, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 10: " + str(model.layers[-1].output_shape))
    
    sgd = SGD(lr = 10r-4, decay = 5e-4, momentum = 0.9, nesterov = False)
    
    model.compile(loss = fcrn_loss, optimizer = sgd, metrics = ['accuracy'])
    
  return model
        
     
    

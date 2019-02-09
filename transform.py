#this file contains functions that transform images;

import tensorflow as tf
import numpy as np
import scipy 

def uniform_random(x_nat, epsilon):
  """Input: batch of images; type: ndarray: size: (batch, 784)
     Output: batch of images with uniform nois; we use clip function 
     to be assured that numbers in matrixs belong to interval (0,1); 
     type: ndarray; size: (batch, 784);	
  """
  x = x_nat + np.random.uniform(-epsilon, epsilon, x_nat.shape) 
  return x

def grando_transform_rotate_batch(batch_of_images, rotate):
  """Input: batch of images; type: ndarray: size: (batch, 784); angle; type: int;
     Output: batch of rotated images; type: ndarray; size: (batch, 784);	
  """
  t = np.array([])
  reshape_batch_of_images = batch_of_images.reshape(len(batch_of_images), 28, 28)
  for i in reshape_batch_of_images:
    t = np.append(t, scipy.ndimage.rotate(i, rotate, reshape = False))
  t = t.reshape(len(batch_of_images), 784)
  return t

def grando_transform_shift_batch(batch_of_images, shift):
  """Input: batch of images; type: ndarray: size: (batch, 784); shift; type: float;
     Output: batch of shifted images; type: ndarray; size: (batch, 784);	
  """
  t = np.array([])
  reshape_batch_of_images = batch_of_images.reshape(len(batch_of_images), 28, 28)
  for i in reshape_batch_of_images:
    t = np.append(t, scipy.ndimage.interpolation.shift(i, float(shift)))
  t = t.reshape(len(batch_of_images), 784)
  return t

def grando_transform_gauss_batch(batch_of_images, mean, variance):
  """Input: batch of images; type: ndarray: size: (batch, 784)
     Output: batch of images with gaussian nois; we use clip function 
     to be assured that numbers in matrixs belong to interval (0,1); 
     type: ndarray; size: (batch, 784);	
  """
  x = batch_of_images + np.random.normal(mean, variance, batch_of_images.shape)
  return x


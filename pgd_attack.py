"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from transform import *


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, bool_uniform, angle, shift, mean, variance, loss_func, bool_rotation, bool_shift, bool_gauss, bool_natural):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = bool_uniform
    self.angle = angle
    self.shift = shift
    self.mean = mean
    self.variance = variance
    self.bool_roration = bool_rotation
    self.bool_shift = bool_shift
    self.bool_gauss = bool_natural
    self.bool_natural = bool_natural
    

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = uniform_random(x_nat, self.epsilon)
    if self.bool_gauss:
      x = grando_transform_gauss_batch(x_nat, self.mean, self.variance)
    if self.bool_rotation:
      x = grando_transform_rotate_batch(x_nat, self.angle)
    if self.bool_shift:
      x = grando_transform_shift(x_nat, self.shift)
    if self.bool_natural:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)
      
      if self.rand:
        x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
        x = np.clip(x, 0, 1)
      if self.bool_gauss:
        x = np.clip(x, x_nat - self.variance, x_nat + self.variance)
        x = np.clip(x, 0, 1)
      if self.bool_shift:
        x = np.clip(x, x_nat - self.shift, x_nat + self.shift)
        x = np.clip(x, 0, 1)
      if self.bool_rotation:
        x = np.clip(x, x_nat - self.angle, x_nat + slef.angle)
        x = np.clip(x, 0, 1)
    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['bool_uniform'],
                         config['angle'],
                         config['shift'],
                         config['mean'],
                         config['variance'],
                         config['loss_func'],
                         config['bool_rotation'],
                         config['bool_shift'],
                         config['bool_gauss'],
                         config['bool_natural'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

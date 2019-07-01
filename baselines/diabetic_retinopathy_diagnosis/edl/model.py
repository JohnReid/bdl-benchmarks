# Copyright 2019 BDL Benchmarks Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definition of the VGGish network for Monte Carlo Dropout baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkm = tfk.models
tfkl = tfk.layers
tfkb = tfk.backend


def relu_evidence(logits):
  """Calculate evidence from logits using a ReLU."""
  return tfkl.Activation("relu")(logits, name='relu_evidence')


def exp_evidence(logits, clip_value=10.0):
  """Calculate evidence from logits using a clipped exp."""
  return tf.exp(tf.clip_by_value(logits, -clip_value, clip_value), name='exp_evidence')


def KL(alpha, K):
  """Calculate the Kullback-Leibler divergence of a Dirichlet(alpha)
  distribution with a Dirichlet(1, ..., 1) distribution.
  """
  beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
  S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
  S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
  lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),
                                           axis=1, keepdims=True)
  lnB_uni = tf.reduce_sum(tf.lgamma(beta), axis=1,
                          keepdims=True) - tf.lgamma(S_beta)
  dg0 = tf.math.digamma(S_alpha)
  dg1 = tf.math.digamma(alpha)

  kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni
  return tf.reduce_mean(kl)


def mse_loss(y, alpha):
  #
  # alpha sums and means
  S = tf.reduce_sum(alpha, axis=1, keepdims=True, name="S")
  print('alpha: ', alpha)
  print('S: ', S)
  phat = tf.divide(alpha, S, name='phat')
  #
  # Error term
  residuals = tf.subtract(y, phat, name='residuals')
  E = tf.square(residuals, name='E')
  #
  # Variance term
  V = tf.multiply(phat, (1 - phat) / (S + 1), name='V')
  #
  return tf.add(tf.reduce_sum(E, axis=1), tf.reduce_sum(V, axis=1), name='mse_loss')


def loss_regulariser(alpha, other=None):
  """Loss regularisation term (without lambda_t)
  """
  if other is None:
    other = tfd.Dirichlet(tf.ones(alpha.shape[-1]))
  regulariser = tfd.Dirichlet(alpha).kl_divergence(other, name='regulariser')
  tf.summary.histogram('regulariser', data=regulariser)
  return regulariser


def annealing_coefficient(epoch):
  """The annealing coefficient grows as the number of epochs to a maximum of 1.
  """
  coef = tf.minimum(1.0, tf.cast(epoch / 10, tf.float32))
  tf.summary.scalar('annealing coefficient', data=coef, step=epoch)
  return coef


def mse_regularised_loss(y, alpha, lambda_t, sample_weight=None):
  regularisation_term = lambda_t * loss_regulariser(alpha)
  mse_term = mse_loss(y, alpha)
  tf.summary.histogram('regularisation term', data=regularisation_term)
  tf.summary.histogram('mse term', data=mse_term)
  # print('y: ', y)
  # print('Regularisation: ', regularisation_term)
  # print('MSE term: ', mse_term)
  return mse_term + regularisation_term


def EDL_loss(func=tf.math.digamma):
  """Evidential deep learning loss."""
  def loss_func(p, alpha, epoch):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1

    A = tf.reduce_mean(tf.reduce_sum(p * (func(S) - func(alpha)), 1, keepdims=True))

    annealing_coef = annealing_coefficient(epoch)

    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp)

    return A + B
  return loss_func


def make_loss(losstype, epoch, EDL_func=tf.math.digamma):
  """
  Make a loss function using given global and annealing steps.
  """
  from functools import partial
  if "mse" == losstype:
    loss = partial(mse_regularised_loss, lambda_t=annealing_coefficient(epoch))
  elif "mse" == losstype:
    loss = partial(EDL_loss(EDL_func), epoch=epoch)
  else:
    raise ValueError('Unknown loss: {}'.format(losstype))
  return loss


def EDL_model(logits_model,
              input_shape,
              learning_rate,
              epoch,
              logits_to_evidence=exp_evidence,
              additional_metrics=[]):
  """Convert logits to alpha

  Args:
    logits: Model to predict logits.
    learning_rate: `float`, ADAM optimizer learning rate.

  Results:
    A compiled model.
  """
  from bdlb.diabetic_retinopathy_diagnosis.benchmark import DiabeticRetinopathyDiagnosisBenchmark

  # Feedforward neural network
  inputs = tfk.Input(shape=input_shape)

  #
  # Calculate the evidence from the logits calculated by the logits model
  evidence = logits_to_evidence(logits_model(inputs))

  #
  # Alpha is the parameter for the Dirichlet, alpha0 is the sum
  alpha = tf.add(evidence, 1, name='alpha')
  alpha0 = tf.reduce_sum(alpha, axis=1, keepdims=True, name='alpha_zero')
  dirichlet = tfd.Dirichlet(alpha)
  entropy = dirichlet.entropy()

  def entropy_mean(y_true, y_pred):
    return tf.reduce_mean(entropy, name='entropy_mean')

  #
  # Calculate the mean probabilities
  p = tf.divide(alpha, alpha0, name='p')

  #
  # The output is the probability of a positive classification
  # outputs = p[:, 1]  # This didn't work
  p_pos = tf.slice(p, [0, 1], [-1, -1], name='p_pos')

  #
  # Create the loss function using function closure of alpha
  def custom_loss(y_true, _):
    print('y_true: ', y_true)
    print('_: ', _)
    # Why must we index y_true? It seems to have shape (None, None). Shouldn't this be (None,) or (None, 1)?
    y_correct_dim = y_true[:, 0]
    # Why must we cast this here? Is there no way to tell Keras that y_true will be int?
    y_true_int = tf.cast(y_correct_dim, tf.int32)
    # Make one-hot for MSE loss
    y_one_hot = tf.one_hot(y_true_int, depth=2, name="y_one_hot")
    # print('y one hot: ', y_one_hot)
    loss = make_loss('mse', epoch=epoch)(y_one_hot, alpha)
    print('Loss: ', loss)
    return loss

  #
  # Compile the model
  print('inputs: ', inputs)
  print('alpha: ', alpha)
  print('p_pos: ', p_pos)
  model = tfkm.Model(inputs=inputs, outputs=[alpha, p_pos])
  metrics = DiabeticRetinopathyDiagnosisBenchmark.metrics()
  metrics.append(entropy_mean)
  metrics += additional_metrics
  model.compile(loss=[None, custom_loss],
                optimizer=tfk.optimizers.Adam(learning_rate),
                metrics=[[], metrics])
  return model


def predict(x, model, type="entropy"):
  """EDL uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space,
      with shape [B, H, W, 3], where B the batch size and
      H, W the input images height and width accordingly.
    model: `tensorflow.keras.Model`, a probabilistic model,
      which accepts input with shape [B, H, W, 3] and
      outputs sigmoid probability [0.0, 1.0], and also
      accepts boolean arguments `training=True` for enabling
      dropout at test time.
    type: (optional) `str`, type of uncertainty returns,
      one of {"entropy"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  #
  # Get shapes of data
  B, _, _, _ = x.shape
  #
  # Forward pass through the model
  alpha, p_pos = model(x)
  #
  # Calculate the expected entropy of a draw from the Dirichlet
  # parameterised by alpha
  exp_H = dirichlet_expected_entropy(alpha)
  #
  # Use predictive entropy for uncertainty
  if type == "entropy":
    uncertainty = exp_H
  else:
    raise ValueError("Unrecognized type={} provided, use one of {'entropy'}".  format(type))
  #
  return p_pos, uncertainty


def categorical_entropy(p):
  """The entropy of a categorical distribution."""
  from scipy.special import xlogy
  return - xlogy(p, p).sum(axis=-1)


def dirichlet_expected_entropy(alpha):
  """The expected entropy of a categorical distribution drawn from Dirichlet(alpha).
  See https://math.stackexchange.com/a/3195376/203036"""
  from scipy.special import digamma
  A = alpha.sum(axis=-1)
  # print(alpha.shape)
  # print(A.shape)
  return digamma(A + 1) - (alpha / np.expand_dims(A, axis=-1) * digamma(alpha + 1)).sum(axis=-1)


def test_dirichlet_expected_entropy():
  """Double check that our calculation of the expected entropy is close to sampled entropies.
  """
  import scipy.stats as st
  N = 9  # Number of distinct p
  M = 10000  # Number of samples
  alpha = st.gamma.rvs(1, size=3 * N).reshape((N, -1))
  exp_H = dirichlet_expected_entropy(alpha)
  p = np.array([st.dirichlet.rvs(a, size=M) for a in alpha])
  p.shape
  H = categorical_entropy(p)
  sample_H = H.mean(axis=-1)
  # print(exp_H)
  # print(sample_H)
  assert(np.isclose(exp_H, sample_H, rtol=0.01, atol=0.01).all())


def VGG_model(dropout_rate,
              num_base_filters,
              l2_reg,
              input_shape,
              output_dim):
  """VGG-like model with dropout for diabetic retinopathy diagnosis.

  Args:
    dropout_rate: `float`, the rate of dropout, between [0.0, 1.0).
    num_base_filters: `int`, number of convolution filters in the
      first layer.
    l2_reg: `float`, the L2-regularization coefficient.
    input_shape: `iterable`, the shape of the images in the input layer.
    output_dim: `int`, the number of outputs

  Returns:
    A tensorflow.keras.Sequential VGG-like model with dropout.
  """
  return tfk.Sequential([
      tfkl.InputLayer(input_shape),
      # Block 1
      tfkl.Conv2D(filters=num_base_filters,
                  kernel_size=3,
                  strides=(2, 2),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 2
      tfkl.Conv2D(filters=num_base_filters,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 3
      tfkl.Conv2D(filters=num_base_filters * 2,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 2,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 4
      tfkl.Conv2D(filters=num_base_filters * 4,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 4,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 4,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 4,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 5
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("relu"),
      # Global poolings
      tfkl.Lambda(lambda x: tfk.backend.concatenate(
          [tfkl.GlobalAvgPool2D()(x),
           tfkl.GlobalMaxPool2D()(x)], axis=1)),
      # Fully-connected - now we need two outputs
      tfkl.Dense(output_dim, kernel_regularizer=tfk.regularizers.l2(l2_reg)),
  ])

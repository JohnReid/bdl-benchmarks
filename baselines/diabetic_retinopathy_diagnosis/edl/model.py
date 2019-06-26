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
  return tfkl.Activation("relu")(logits)


def exp_evidence(logits, clip_value=10.0):
  """Calculate evidence from logits using a clipped exp."""
  return tf.exp(tf.clip_by_value(logits, -clip_value, clip_value))


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
  S = tf.reduce_sum(alpha, axis=1, name="S")
  phat = alpha / S
  #
  # Error term
  E = (y - phat)**2
  #
  # Variance term
  V = phat * (1 - phat) / (S + 1)
  #
  return tf.reduce_sum(E, axis=1) + tf.reduce_sum(V, axis=1)


def loss_regulariser(alpha, other=None):
  """Loss regularisation term (without lambda_t)
  """
  if other is None:
    other = tfd.Dirichlet(tf.ones(alpha.shape[-1]))
  return tfd.Dirichlet(alpha).kl_divergence(other)


def annealing_coefficient(epoch):
  """The annealing coefficient grows as the number of epochs to a maximum of 1.
  """
  return tf.minimum(1.0, tf.cast(epoch / 10, tf.float32))


def mse_regularised_loss(y, alpha, lambda_t, sample_weight=None):
  regularisation_term = lambda_t * loss_regulariser(alpha)
  mse_term = mse_loss(y, alpha)
  print('y: ', y)
  print('Regularisation: ', regularisation_term)
  print('MSE term: ', mse_term)
  return mse_term + regularisation_term


def EDL_loss(func=tf.math.digamma):
  """Evidential deep learning loss."""
  def loss_func(p, alpha, global_step, epoch):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1

    A = tf.reduce_mean(tf.reduce_sum(p * (func(S) - func(alpha)), 1, keepdims=True))

    annealing_coef = tf.minimum(1.0, tf.cast(epoch / 10, tf.float32))

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


def create_VGG_like(dropout_rate,
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


def wrap_EDL_model(logits,
                   input_shape,
                   learning_rate,
                   global_step,
                   epoch,
                   logits_to_evidence=exp_evidence):
  """Wrap a model with EDL loss.

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
  # Calculate the evidence from the logits
  evidence = logits_to_evidence(logits(inputs))

  #
  # Alpha is the parameter for the Dirichlet
  alpha = tf.add(evidence, 1, name='alpha')

  #
  # Calculate the mean probabilities
  p = tf.divide(alpha, tf.reduce_sum(alpha, axis=1, keepdims=True), name='p')

  #
  # The output is the probability of a positive classification
  # outputs = p[:, 1]  # This didn't work
  outputs = tf.slice(p, [0, 1], [-1, -1], name='outputs')

  #
  # Create the loss function from Murat's code above
  def custom_loss(y, _):
    print('y: ', y)
    y_one_hot = tf.one_hot(tf.cast(y, tf.int32), depth=2, name="y_one_hot")
    print('y one hot: ', y_one_hot)
    loss = make_loss('mse', epoch=epoch)(y_one_hot, alpha)
    print('Loss: ', loss)
    return loss

  #
  # Compile the model
  model = tfkm.Model(inputs=inputs, outputs=outputs)
  model.compile(loss=custom_loss,
                optimizer=tfk.optimizers.Adam(learning_rate),
                metrics=DiabeticRetinopathyDiagnosisBenchmark.metrics())

  return model

  # # Calculate accuracy
  # pred = tf.argmax(logits, 1)
  # truth = tf.argmax(Y, 1)
  # match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
  # acc = tf.reduce_mean(match)

  # total_evidence = tf.reduce_sum(evidence,1, keepdims=True)
  # mean_ev = tf.reduce_mean(total_evidence)
  # mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
  # mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20)

  # return g, step, X, Y, epoch, keep_prob, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail


def predict(x, model, num_samples, type="entropy"):
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
      num_samples: `int`, number of Monte Carlo samples
        (i.e. forward passes from dropout) used for
        the calculation of predictive mean and uncertainty.
      type: (optional) `str`, type of uncertainty returns,
        one of {"entropy", "stddev"}.

    Returns:
      mean: `numpy.ndarray`, predictive mean, with shape [B].
      uncertainty: `numpy.ndarray`, uncertainty in prediction,
        with shape [B].
    """
    import scipy.stats

    # Get shapes of data
    B, _, _, _ = x.shape

    # Monte Carlo samples from different dropout mask at test time
    mc_samples = np.asarray([model(x, training=True)
                             for _ in range(num_samples)]).reshape(-1, B)

    # Bernoulli output distribution
    dist = scipy.stats.bernoulli(mc_samples.mean(axis=0))

    # Predictive mean calculation
    mean = dist.mean()

    # Use predictive entropy for uncertainty
    if type == "entropy":
        uncertainty = dist.entropy()
    # Use predictive standard deviation for uncertainty
    elif type == "stddev":
        uncertainty = dist.std()
    else:
        raise ValueError(
            "Unrecognized type={} provided, use one of {'entropy', 'stddev'}".
            format(type))

    return mean, uncertainty

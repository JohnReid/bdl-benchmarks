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
"""Uncertainty estimator for the deterministic deep model baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from baselines.diabetic_retinopathy_diagnosis.myEDL.edl_utils import loss, metrics


def VGGDrop(dropout_rate, num_base_filters, learning_rate, l2_reg, input_shape):
  """VGG-like model with dropout for diabetic retinopathy diagnosis.

  Args:
    dropout_rate: `float`, the rate of dropout, between [0.0, 1.0).
    num_base_filters: `int`, number of convolution filters in the
      first layer.
    learning_rate: `float`, ADAM optimizer learning rate.
    l2_reg: `float`, the L2-regularization coefficient.
    input_shape: `iterable`, the shape of the images in the input layer.

  Returns:
    A tensorflow.keras.Sequential VGG-like model with dropout.
  """
  import tensorflow as tf
  tfk = tf.keras
  tfkl = tfk.layers
  from bdlb.diabetic_retinopathy_diagnosis.benchmark import DiabeticRetinopathyDiagnosisBenchmark

  # Feedforward neural network
  model = tfk.Sequential([
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
    # Fully-connected
    tfkl.Dense(2, kernel_regularizer=tfk.regularizers.l2(l2_reg)),
    #tfkl.Activation("sigmoid")
  ])
  model.global_step = tf.Variable(tf.constant(0), trainable=False, name='global_step')


  model.compile(loss= loss(),
                optimizer=tfk.optimizers.Adam(learning_rate),
                metrics=metrics())
  DiabeticRetinopathyDiagnosisBenchmark.metrics()

  return model

def predict(x, model, num_samples, type="entropy"):
  """Simple sigmoid uncertainty estimator.
    
  Args:
    x: `numpy.ndarray`, datapoints from input space,
      with shape [B, H, W, 3], where B the batch size and
      H, W the input images height and width accordingly.
    model: `tensorflow.keras.Model`, a probabilistic model,
      which accepts input with shape [B, H, W, 3] and
      outputs sigmoid probability [0.0, 1.0], and also
      accepts boolean arguments `training=False` for
      disabling dropout at test time.
    type: (optional) `str`, type of uncertainty returns,
      one of {"entropy", "stddev"}.
  
  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, ncertainty in prediction,
      with shape [B].
  """
  import numpy as np
  import scipy.stats

  # Get shapes of data
  B, _, _, _ = x.shape

  # Single forward pass from the deterministic model
  p = model(x, training=False)

  # Bernoulli output distribution
  dist = scipy.stats.bernoulli(p)

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

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
"""Model definition of the VGGish network for temperature scaling baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bdlb.diabetic_retinopathy_diagnosis.benchmark import DiabeticRetinopathyDiagnosisBenchmark
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tfk.layers


def VGG(dropout_rate, num_base_filters, learning_rate, l2_reg, input_shape):
  """VGG-like model for diabetic retinopathy diagnosis.

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
      tfkl.Dense(1, kernel_regularizer=tfk.regularizers.l2(l2_reg)),
      tfkl.Activation("sigmoid")
  ])

  model.compile(loss=DiabeticRetinopathyDiagnosisBenchmark.loss(),
                optimizer=tfk.optimizers.Adam(learning_rate),
                metrics=DiabeticRetinopathyDiagnosisBenchmark.metrics())

  return model


def predict(x, model, type="entropy"):
  """Temperature scaled uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space,
      with shape [B, H, W, 3], where B the batch size and
      H, W the input images height and width accordingly.
    model: `tensorflow.keras.Model`, a probabilistic model,
      which accepts input with shape [B, H, W, 3] and
      outputs logits with shape [B, 2].
    type: (optional) `str`, type of uncertainty returns,
      one of {"entropy", "stddev"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  from scipy.stats import bernoulli
  #
  logits = model(x, training=False)
  p = tf.nn.softmax(logits)[:, 1]
  #
  # Bernoulli output distribution
  dist = bernoulli(p)
  #
  # Predictive mean calculation
  mean = dist.mean()
  #
  # Choose uncertainty measure
  if type == "entropy":
    # Use predictive entropy for uncertainty
    uncertainty = dist.entropy()
  elif type == "stddev":
    # Use predictive standard deviation for uncertainty
    uncertainty = dist.std()
  else:
    raise ValueError(
        "Unrecognized type={} provided, use one of {'entropy', 'stddev'}".
        format(type))
  #
  return mean, uncertainty


class BinaryProbToMulticlass(tfkl.Layer):
  """
  A Keras layer that takes binary classification probabilities and converts them to
  2-way multiclass probabilities.
  """

  def __init__(self):
    super(BinaryProbToMulticlass, self).__init__()

  def build(self, input_shape):
    super(BinaryProbToMulticlass, self).build(input_shape)  # Be sure to call this at the end

  def compute_output_shape(self, input_shape):
    print('input_shape: ', input_shape)
    return input_shape + (2,)

  def call(self, p_pos):
    p_neg = 1 - p_pos
    result = tf.concat([p_neg, p_pos], axis=-1, name='multiclass')
    # print('p_pos: ', p_pos.shape)
    # print('result: ', result.shape)
    return result


class TemperatureScaling(tfkl.Layer):
  """
  A Keras layer that wraps a model with temperature scaling.
  Note that the inputs should be the classification logits, not the softmax or log softmax.
  """

  def __init__(self):
    super(TemperatureScaling, self).__init__()
    self.temperature = tf.Variable(initial_value=[1.5])
    # print('temperature: ', self.temperature.shape)

  def build(self, input_shape):
    super(TemperatureScaling, self).build(input_shape)  # Be sure to call this at the end

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, logits):
    """
    Perform temperature scaling on logits
    """
    return tf.divide(logits, self.temperature, name='scaled')

  def create_objective(self, y_true, logits):
    """
    Create an objective function to use to optimise the temperature.

    Args:
      y_true: the ground truth labels
      logits: the logits

    Returns:
      An objective function that returns (NLL, dNLL / dTemp) for a given temperature
    """
    def objective(temperature):
      self.temperature.assign(temperature)
      #
      # Don't calculate gradients for any variables except for temperature
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.temperature)
        scaled_logits = self.call(logits)
        loss_per_sample = tfk.losses.categorical_crossentropy(y_true, scaled_logits, from_logits=True)
        loss = tf.reduce_sum(loss_per_sample, name='loss')
      grads = tape.gradient(loss, self.temperature)
      # print('loss: ', loss.shape)
      # print('grads: ', grads.shape)
      return loss, grads

    return objective

  def optimise_temperature(self, y_true, logits):
    """
    Tune the temperature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    #
    # Check the calibration error before optimising
    try:
      import sail.metrics as me
      ce = me.GPleissCalibrationError()
      accuracy, confidence = me.accuracy_and_confidence(tf.argmax(y_true, axis=-1).numpy(), logits.numpy())
      error_before = ce.error(*ce.frequencies(accuracy, confidence))
      print('Error before calibration: ', error_before)
    except ImportError:
      import warnings
      warnings.warn('Could not import sail.metrics.')
    #
    # Create the objective function
    objective = self.create_objective(y_true, logits)
    #
    # Run the optimisation
    results = tfp.optimizer.bfgs_minimize(objective, self.temperature, max_iterations=50)
    # print(dir(results))
    #
    # Check the results
    if not results.converged:
      import warnings
      warnings.warn('LBFGS did not converge')
      print(results)
    #
    # Update the temperature
    new_temperature = results.position
    print('Optimised temperature: {}'.format(new_temperature))
    self.temperature.assign(new_temperature)
    #
    # Check the calibration error after optimising
    try:
      accuracy, confidence = me.accuracy_and_confidence(tf.argmax(y_true, axis=-1).numpy(),
                                                        logits.numpy() / new_temperature.numpy())
      error_after = ce.error(*ce.frequencies(accuracy, confidence))
      print('Error after calibration: ', error_after)
      if error_after > error_before:
        import warnings
        warnings.warn('Temperature scaling increased the calibration error!')
        raise ValueError('Temperature scaling increased the calibration error!')
    except NameError:
      import warnings
      warnings.warn('Could not calculate calibration error.')
    #
    return self

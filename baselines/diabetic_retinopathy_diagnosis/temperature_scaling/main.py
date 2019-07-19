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
"""Script for training and evaluating temperature scaling baseline for
Diabetic Retinopathy Diagnosis benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools
import datetime

import bdlb
# from bdlb.core import plotting
from baselines.diabetic_retinopathy_diagnosis.temperature_scaling import model

from absl import app
from absl import flags
import tensorflow as tf
tf.__version__
tfk = tf.keras
tfkl = tf.keras.layers
tfkm = tfk.models

bdlb.tf_limit_memory_growth()

##########################
# Command line arguments #
##########################
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="output_dir",
    default='output',
    help="Path to store model, tensorboard and report outputs.",
)
flags.DEFINE_string(
    name="model_dir",
    default='output/Deterministic/20190717-092955/checkpoints/',
    help="Path to load model weights from.",
)
flags.DEFINE_enum(
    name="level",
    default="medium",
    enum_values=["realworld", "medium"],
    help="Downstream task level, one of {'medium', 'realworld'}.",
)
flags.DEFINE_integer(
    name="batch_size",
    default=128,
    help="Batch size used for training.",
)
flags.DEFINE_integer(
    name="max_batches",
    default=0,
    help="Maximum number of batches used for training in each epoch.",
)
flags.DEFINE_integer(
    name="num_epochs",
    default=50,
    help="Number of epochs of training over the whole training set.",
)
flags.DEFINE_enum(
    name="uncertainty",
    default="entropy",
    enum_values=["stddev", "entropy"],
    help="Uncertainty type, one of those defined "
    "with `estimator` function.",
)
flags.DEFINE_integer(
    name="num_base_filters",
    default=32,
    help="Number of base filters in convolutional layers.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=4e-4,
    help="ADAM optimizer learning rate.",
)
flags.DEFINE_float(
    name="dropout_rate",
    default=0.1,
    help="The rate of dropout, between [0.0, 1.0).",
)
flags.DEFINE_float(
    name="l2_reg",
    default=5e-5,
    help="The L2-regularization coefficient.",
)


def main(argv):

  # print(argv)
  # print(FLAGS)

  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  out_dir = os.path.join(FLAGS.output_dir, 'TS', current_time)

  ##########################
  # Hyperparmeters & Model #
  ##########################
  input_shape = dict(medium=(256, 256, 3), realworld=(512, 512, 3))[FLAGS.level]

  hparams = dict(dropout_rate=FLAGS.dropout_rate,
                 num_base_filters=FLAGS.num_base_filters,
                 learning_rate=FLAGS.learning_rate,
                 l2_reg=FLAGS.l2_reg,
                 input_shape=input_shape)
  classifier = model.VGG(**hparams)
  # classifier.summary()
  print('********** Output dir: {} ************'.format(out_dir))

  latest = tf.train.latest_checkpoint(FLAGS.model_dir)
  print('********** Loading checkpoint weights: {}'.format(latest))
  classifier.load_weights(latest)

  #############
  # Load Task #
  #############
  dtask = bdlb.load(
      benchmark="diabetic_retinopathy_diagnosis",
      level=FLAGS.level,
      batch_size=FLAGS.batch_size,
      download_and_prepare=False,  # do not download data from this script
  )
  ds_train, ds_validation, ds_test = dtask.datasets

  ###############
  # Build model #
  ###############
  binary_prob_to_multi = model.BinaryProbToMulticlass()
  inv_softmax = model.InverseSoftmaxFixedMean(mean=1.)
  ts_layer = model.TemperatureScaling()
  multi_to_binary_prob = model.MulticlassToBinaryProb()
  ts_model = tfkm.Sequential([classifier,
                              binary_prob_to_multi,
                              inv_softmax,
                              ts_layer,
                              tfkl.Softmax(),
                              multi_to_binary_prob])
  ts_model.build(input_shape=hparams['input_shape'])
  classifier.summary()
  ts_model.summary()
  print('classifier: ', classifier.get_layer(1).output_shape)
  print('ts_model: ', ts_model.get_layer(1).output_shape)

  ########################
  # Optimise temperature #
  ########################
  #
  # Calculate logits for validation set
  y_true = []
  logits = []
  for x, y in ds_validation:
    p = classifier(x)
    p_multi = binary_prob_to_multi(p)
    logit = inv_softmax(p_multi)
    logits.append(logit)
    y_true.append(tf.one_hot(y, depth=2))
  y_true = tf.concat(y_true, axis=0)
  logits = tf.concat(logits, axis=0)
  #
  # Optimise temperature
  ts_layer.optimise_temperature(y_true, logits)

  ##############
  # Evaluation #
  ##############
  additional_metrics = []
  try:
    import sail.metrics
    additional_metrics.append(('ECE', sail.metrics.GPleissCalibrationError()))
  except ImportError:
    import warnings
    warnings.warn('Could not import SAIL metrics.')
  dtask.evaluate(functools.partial(model.predict, model=ts_model, type=FLAGS.uncertainty),
                 dataset=ds_test,
                 output_dir=os.path.join(out_dir, 'evaluation'),
                 additional_metrics=additional_metrics)
  print('Temperature: ', ts_layer.temperature.value())


if __name__ == "__main__":
  import sys
  sys.argv = sys.argv[:1]
  app.run(main)

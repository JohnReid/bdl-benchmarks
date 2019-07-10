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

import sail.metrics
import bdlb
# from bdlb.core import plotting
from baselines.diabetic_retinopathy_diagnosis.temperature_scaling import model

from absl import app
from absl import flags
import tensorflow as tf
tf.__version__
tfk = tf.keras
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
flags.DEFINE_integer(
    name="num_mc_samples",
    default=10,
    help="Number of Monte Carlo samples used for uncertainty estimation.",
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

  print(argv)
  print(FLAGS)

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
  classifier.summary()
  print('********** Output dir: {} ************'.format(out_dir))

  CHKPT_DIR = 'output/Deterministic/20190705-170745/checkpoints'
  latest = tf.train.latest_checkpoint(CHKPT_DIR)
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

  #################
  # Training Loop #
  #################
  # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # if FLAGS.max_batches > 0:
  #   train_data = ds_train.take(FLAGS.max_batches)
  # else:
  #   train_data = ds_train
  # history = classifier.fit(
  #     train_data,
  #     epochs=FLAGS.num_epochs,
  #     validation_data=ds_validation,
  #     class_weight=dtask.class_weight(),
  #     callbacks=[
  #         tfk.callbacks.TensorBoard(
  #             log_dir=os.path.join(out_dir, 'fit'),
  #             update_freq="epoch",
  #             write_graph=True,
  #             histogram_freq=1,
  #         ),
  #         tfk.callbacks.ModelCheckpoint(
  #             filepath=os.path.join(out_dir, "checkpoints", "weights-{epoch}.ckpt"),
  #             verbose=1,
  #             save_weights_only=True,
  #         )
  #     ],
  # )
  # plotting.tfk_history(history, output_dir=os.path.join(out_dir, "history"))

  ########################
  # Optimise temperature #
  ########################
  ts_layer = model.TemperatureScaling()
  ts_model = tfkm.Sequential([classifier, model.BinaryProbToMulticlass(), ts_layer])
  #
  # Calculate logits for test set
  y_true = []
  logits = []
  for x, y in ds_validation:
    p = classifier(x)
    logit = tf.math.log(p) - tf.math.log(1 - p)
    logit_multi = tf.concat([tf.zeros_like(logit), logit], axis=1)
    logits.append(logit_multi)
    y_true.append(tf.one_hot(y, depth=2))
  y_true = tf.concat(y_true, axis=0)
  logits = tf.concat(logits, axis=0)
  #
  # Optimise temperature
  ts_layer.optimise_temperature(y_true, logits)

  ##############
  # Evaluation #
  ##############
  dtask.evaluate(functools.partial(model.predict,
                                   model=ts_model,
                                   type=FLAGS.uncertainty),
                 dataset=ds_test,
                 output_dir=os.path.join(out_dir, 'evaluation'),
                 additional_metrics=[('ECE', sail.metrics.TFExpectedCalibrationError())])
  print('Temperature: ', ts_layer.temperature.value)

  # Check beta distribution
  # import numpy as np
  # import scipy.stats as stats
  # import pylab as pl
  # p = np.linspace(0, 1, num=101)
  # pl.plot(p, stats.beta.pdf(p, a=2, b=5 / 4), '-o')
  # pl.savefig('beta.png')
  # # !eog beta.png


if __name__ == "__main__":
  import sys
  sys.argv = sys.argv[:1]
  app.run(main)

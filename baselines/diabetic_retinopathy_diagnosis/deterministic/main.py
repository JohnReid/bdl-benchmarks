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
"""Script for training and evaluating a deterministic baseline for
Diabetic Retinopathy Diagnosis benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools
import datetime

import bdlb
from bdlb.core import plotting
from baselines.diabetic_retinopathy_diagnosis.mc_dropout.model import VGGDrop
from baselines.diabetic_retinopathy_diagnosis.deterministic.model import predict

from absl import app
from absl import flags
import tensorflow as tf
tfk = tf.keras

bdlb.tf_limit_memory_growth()

##########################
# Command line arguments #
##########################
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="output_dir",
    default="output",
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
  out_dir = os.path.join(FLAGS.output_dir, 'Deterministic', current_time)

  ##########################
  # Hyperparmeters & Model #
  ##########################
  input_shape = dict(medium=(256, 256, 3), realworld=(512, 512, 3))[FLAGS.level]

  hparams = dict(dropout_rate=FLAGS.dropout_rate,
                 num_base_filters=FLAGS.num_base_filters,
                 learning_rate=FLAGS.learning_rate,
                 l2_reg=FLAGS.l2_reg,
                 input_shape=input_shape)
  classifier = VGGDrop(**hparams)
  # classifier.summary()

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
  history = classifier.fit(
      ds_train,
      epochs=FLAGS.num_epochs,
      validation_data=ds_validation,
      class_weight=dtask.class_weight(),
      callbacks=[
          tfk.callbacks.TensorBoard(
              log_dir=os.path.join(out_dir, "tensorboard"),
              update_freq="epoch",
              write_graph=True,
              histogram_freq=1,
          ),
          tfk.callbacks.ModelCheckpoint(
              filepath=os.path.join(
                  out_dir,
                  "checkpoints",
                  "weights-{epoch}.ckpt",
              ),
              verbose=1,
              save_weights_only=True,
          )
      ],
  )
  plotting.tfk_history(history,
                       output_dir=os.path.join(out_dir, "history"))

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
  dtask.evaluate(functools.partial(predict, model=classifier, type=FLAGS.uncertainty),
                 dataset=ds_test,
                 output_dir=os.path.join(out_dir, 'evaluation'),
                 additional_metrics=additional_metrics)


if __name__ == "__main__":
  app.run(main)

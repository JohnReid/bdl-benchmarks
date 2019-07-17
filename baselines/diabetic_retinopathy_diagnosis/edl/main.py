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
"""Script for training and evaluating evidential deep learning baseline for
Diabetic Retinopathy Diagnosis benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import matplotlib.pyplot as plt

import bdlb
# from bdlb.diabetic_retinopathy_diagnosis.benchmark import DiabeticRetinopathyDiagnosisBenchmark
# from bdlb.core import plotting
import baselines.diabetic_retinopathy_diagnosis.edl.model as model

from absl import app
from absl import flags
import tensorflow as tf
tfk = tf.keras
tfkb = tfk.backend

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

#
# Following advice here to limit memory growth on TF 2.0
# https://www.tensorflow.org/beta/guide/using_gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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

  #
  # Logging / tensorboard
  #
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  out_dir = os.path.join(FLAGS.output_dir, 'EDL', current_time)
  file_writer = tf.summary.create_file_writer(os.path.join(out_dir, "tensorboard"))
  file_writer.set_as_default()

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
  assert isinstance(ds_train, tf.data.Dataset)

  ##########################
  # Hyperparmeters & Model #
  ##########################
  input_shape = dict(medium=(256, 256, 3), realworld=(512, 512, 3))[FLAGS.level]
  global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)
  logits_model = model.VGG_model(dropout_rate=FLAGS.dropout_rate,
                                 num_base_filters=FLAGS.num_base_filters,
                                 l2_reg=FLAGS.l2_reg,
                                 input_shape=input_shape,
                                 output_dim=2)
  print('********** Output dir: {} ************'.format(out_dir))

  # Create optimiser
  optimizer = tfk.optimizers.Adam(FLAGS.learning_rate)

  #################
  # Training Loop #
  #################
  #
  # keep results for plotting
  train_loss_results = []
  train_accuracy_results = []
  test_accuracy_results = []
  #
  # Keep track of statistics
  train_loss_avg = tfk.metrics.Mean()
  train_entropy_avg = tfk.metrics.Mean()
  train_accuracy = tfk.metrics.Accuracy()
  test_accuracy = tfk.metrics.Accuracy()
  #
  # Set up checkpointing
  checkpoint_dir = os.path.join(out_dir, 'checkpoints')
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  root = tf.train.Checkpoint(optimizer=optimizer, model=logits_model, optimizer_step=global_step)

  #
  for epoch in range(FLAGS.num_epochs):
    #
    # Reset statistics
    train_loss_avg.reset_states()
    train_entropy_avg.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    #
    # Calculate annealing coefficient
    lambda_t = model.annealing_coefficient(epoch)
    #
    # Make summaries
    tf.summary.scalar('lambda_t', lambda_t)
    # tf.summary.scalar('global_step', global_step)
    #
    # Training loop: for each batch
    for batch, (x, y) in enumerate(ds_train.take(10)):
      # print('x: ', x.shape)
      # print('y: ', y.shape)
      y_one_hot = tf.one_hot(y, depth=2, name="y_one_hot")  # Make one-hot for MSE loss
      # print('y one hot: ', y_one_hot.shape)
      #
      # Calculate the loss
      with tf.GradientTape() as tape:
        #
        # Calculate the logits and the evidence
        logits = logits_model(x)
        evidence, alpha, alpha0, p_mean, p_pos = model.EDL_model(logits)
        #
        # Calculate the loss
        alpha_mod = (1 - y_one_hot) * evidence + 1
        regularisation_term = lambda_t * model.loss_regulariser(alpha_mod)
        mse_term = model.mse_loss(y_one_hot, alpha)
        loss = mse_term + regularisation_term
        # loss = mse_term
        # print('Loss: ', loss.shape)
        #
        # Calculate gradients
        grads = tape.gradient(loss, logits_model.trainable_variables)
      #
      # Apply gradients
      optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))
      #
      # Calculate the expected entropy
      exp_entropy = model.tf_dirichlet_expected_entropy(alpha)
      #
      # compare predicted label to actual label
      prediction = tf.argmax(logits, axis=1, output_type=y.dtype)
      #
      # Make summaries
      tf.summary.histogram('alpha', alpha)
      tf.summary.histogram('expected_entropy', exp_entropy)
      tf.summary.histogram('regularisation', regularisation_term)
      tf.summary.histogram('mse', mse_term)
      tf.summary.histogram('loss', loss)
      #
      # Track progress
      train_accuracy.update_state(prediction, y)
      train_entropy_avg.update_state(exp_entropy)
      train_loss_avg.update_state(loss)  # add current batch loss
    #
    # Evaluate on test accuracy
    # for test_batch, (x, y) in enumerate(ds_test):
    #   logits = logits_model(x)
    #   prediction = tf.argmax(logits, axis=1, output_type=y.dtype)
    #   # print('y: ', y.shape)
    #   # print('prediction: ', prediction.shape)
    #   test_accuracy.update_state(prediction, y)
    #
    # Log statistics
    if epoch % 1 == 0:
      template = "Epoch {:03d}: Train loss: {:.3f}, Train entropy: {:.3f}, " \
          "Train accuracy: {:.3%}, Test accuracy: {:.3%}"
      print(template.format(epoch,
                            train_loss_avg.result(),
                            train_entropy_avg.result(),
                            train_accuracy.result(),
                            test_accuracy.result()))
    #
    # Make a checkpoint
    if epoch % 1 == 0:
      root.save(checkpoint_prefix)
    # root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    #
    # end epoch
    tf.summary.scalar('train_loss_avg', train_loss_avg.result())
    tf.summary.scalar('train_entropy_avg', train_entropy_avg.result())
    tf.summary.scalar('train_accuracy', train_accuracy.result())
    tf.summary.scalar('test_accuracy', test_accuracy.result())
    train_loss_results.append(train_loss_avg.result())
    train_loss_results.append(train_entropy_avg.result())
    train_accuracy_results.append(train_accuracy.result())
    test_accuracy_results.append(test_accuracy.result())
    global_step.assign_add(1)
  #
  # Make a final checkpoint
  root.save(checkpoint_prefix)

  # history = classifier.fit(
  #     ds_train,
  #     epochs=FLAGS.num_epochs,
  #     validation_data=ds_validation,
  #     class_weight=dtask.class_weight(),
  #     callbacks=[
  #         tfk.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: increment_epoch()),
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

  #
  # Plot history
  fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
  fig.suptitle('Training Metrics')
  #
  axes[0].set_ylabel("Loss", fontsize=14)
  axes[0].plot(train_loss_results)
  #
  axes[1].set_ylabel("Accuracy", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=14)
  axes[1].plot(train_accuracy_results, label='train')
  axes[1].plot(test_accuracy_results, label='test')
  axes[1].legend(loc='lower right')
  plt.savefig('tmp.png')

  ##############
  # Evaluation #
  ##############
  # dtask.evaluate(functools.partial(model.predict,
  #                                  model=edl_model,
  #                                  type=FLAGS.uncertainty),
  #                dataset=ds_test,
  #                output_dir=os.path.join(out_dir, 'evaluation'))


if __name__ == "__main__":
  import sys
  sys.argv = sys.argv[:1]
  app.run(main)

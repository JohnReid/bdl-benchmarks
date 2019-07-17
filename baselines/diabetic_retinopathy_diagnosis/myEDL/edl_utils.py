import tensorflow as tf
import numpy as np


def relu_evidence(logits):
  """Calculate evidence from logits using a ReLU."""
  return tf.nn.relu(logits)


def exp_evidence(logits, smooth=10.0):
  """Calculate evidence from logits using exp."""
  return tf.exp(logits/smooth)


def softplus_evidence(logits):
  """Calculate evidence from logits using softplus."""
  return tf.nn.softplus(logits)

def get_pub(logits, logits2evidence=exp_evidence):
    ev = logits2evidence(logits)
    alpha = ev + 1
    tot_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
    b = ev / tot_alpha
    p = alpha / tot_alpha
    u = 2.0 / tot_alpha
    return alpha, p, u, b


def KL(alpha):
  """Calculate the Kullback-Leibler divergence of a Dirichlet(alpha)
  distribution with a Dirichlet(1, ..., 1) distribution.
  """
  beta = tf.ones_like(alpha)
  S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
  S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
  lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),
                                                axis=1, keepdims=True)
  lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=1,
                          keepdims=True) - tf.math.lgamma(S_beta)
  dg0 = tf.math.digamma(S_alpha)
  dg1 = tf.math.digamma(alpha)

  kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni
  return tf.reduce_mean(kl)

def mse_loss(y, alpha, annealing_coef=0.1):
    E = alpha - 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    m = alpha / S

    A = tf.reduce_sum((y - m) ** 2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)

    alp = E * (1 - y) + 1
    C = annealing_coef * KL(alp)
    return (A + B) + C

def EDL_loss(func=tf.math.digamma):
  """Evidential deep learning loss."""
  def loss_func(y, alpha, annealing_coef=0.1):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1

    A = tf.reduce_sum(y * (func(S) - func(alpha)), 1, keepdims=True)

    alp = E * (1 - y) + 1
    B = annealing_coef * KL(alp)

    return A + B
  return loss_func


def loss(global_step=5000, annealing_step=50000):
  def my_loss(y, logits):
      alpha, p, u, b = get_pub(logits)
      y_hot = tf.one_hot(tf.cast(y[:,0],tf.int32),depth=2)

      coef = 0.1#tf.minimum(1.0, tf.cast(global_step, tf.float32) / annealing_step)

      return mse_loss(y_hot, alpha, coef)
  return my_loss

def categorical_accuracy(y_true, y_pred):
    if len(y_true.get_shape().as_list()) > 1:
        y_true = tf.cast(y_true[:,0], tf.int64)
    acc = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1)), tf.float32)
    return acc

def binary_accuracy(y_true, y_pred):
    _, p,u,b=get_pub(y_pred)
    y_pred = p[:, 1:2]
    y_true = y_true[:, 0:1]
    return tf.keras.metrics.BinaryAccuracy()(y_true, y_pred)

# AUC for a binary classifier
def auc_metric(y_true, y_pred):
    _, p, u, b = get_pub(y_pred)
    y_pred = p[:, 1]
    y_true = y_true[:, 0]
    return tf.keras.metrics.AUC()(y_true, y_pred)

def metrics():
    """Evaluation metrics used for monitoring training."""
    return [binary_accuracy, auc_metric]




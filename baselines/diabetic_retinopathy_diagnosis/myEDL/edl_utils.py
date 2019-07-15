import tensorflow as tf
import numpy as np

def annealing_coefficient(epoch):
  """The annealing coefficient grows as the number of epochs to a maximum of 1.
  """
  coef = tf.minimum(1.0, tf.cast(epoch, tf.float32) / 10)
  return coef


def relu_evidence(logits):
  """Calculate evidence from logits using a ReLU."""
  return tf.nn.relu(logits)


def exp_evidence(logits, clip_value=10.0):
  """Calculate evidence from logits using a clipped exp."""
  return tf.exp(logits/10, name='exp_evidence')


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

def mse_loss(y, alpha):
    E = alpha - 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    m = alpha / S

    A = tf.reduce_sum((y - m) ** 2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True)

    annealing_coef = 0.1 #annealing_coefficient(epoch)

    alp = E * (1 - y) + 1
    C = annealing_coef * KL(alp)
    return (A + B) + C

def EDL_loss(func=tf.math.digamma):
  """Evidential deep learning loss."""
  def loss_func(y, alpha):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1

    A = tf.reduce_sum(y * (func(S) - func(alpha)), 1, keepdims=True)

    annealing_coef = 0.1#annealing_coefficient(epoch)

    alp = E * (1 - y) + 1
    B = annealing_coef * KL(alp)

    return A + B
  return loss_func


def loss():
  def my_loss(y, logits):
      evidence = exp_evidence(logits)
      alpha = evidence + 1
      y_hot = tf.one_hot(tf.cast(y[:,0],tf.int32),depth=2)
      return mse_loss(y_hot, alpha)
  return my_loss

def loss1():
  def my_loss(y, logits):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
  return my_loss


def categorical_accuracy(y_true, y_pred):
    if len(y_true.get_shape().as_list()) > 1:
        y_true = tf.cast(y_true[:,0], tf.int64)
    acc = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1)), tf.float32)
    return acc

# AUC for a binary classifier
def auc_metric(y_true, y_pred):
    ev = exp_evidence(y_pred)
    alpha = ev + 1
    p = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
    y_pred = p[:, 1]
    y_true = y_true[:, 0]
    return tf.keras.metrics.AUC()(y_true, y_pred)

def metrics():
    """Evaluation metrics used for monitoring training."""
    import tensorflow as tf
    tfk = tf.keras
    return [categorical_accuracy, auc_metric] #[tfk.metrics.CategoricalAccuracy(), tfk.metrics.AUC()]




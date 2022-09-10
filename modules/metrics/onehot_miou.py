#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops, math_ops, array_ops
from tensorflow.keras.metrics import MeanIoU
from utils import print_np


# class OneHotMeanIoU(Metric):
#   """Computes mean Intersection-Over-Union metric for one-hot encoded labels.
#   General definition and computation:
#   Intersection-Over-Union is a common evaluation metric for semantic image
#   segmentation.
#   For an individual class, the IoU metric is defined as follows:
#   ```
#   iou = true_positives / (true_positives + false_positives + false_negatives)
#   ```
#   To compute IoUs, the predictions are accumulated in a confusion matrix,
#   weighted by `sample_weight` and the metric is then calculated from it.
#   If `sample_weight` is `None`, weights default to 1.
#   Use `sample_weight` of 0 to mask values.
#   This class can be used to compute the mean IoU for multi-class classification
#   tasks where the labels are one-hot encoded (the last axis should have one
#   dimension per class). Note that the predictions should also have the same
#   shape. To compute the mean IoU, first the labels and predictions are converted
#   back into integer format by taking the argmax over the class axis. Then the
#   same computation steps as for the base `MeanIoU` class apply.
#   Note, if there is only one channel in the labels and predictions, this class
#   is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.
#   Also, make sure that `num_classes` is equal to the number of classes in the
#   data, to avoid a "labels out of bound" error when the confusion matrix is
#   computed.
#   Args:
#     num_classes: The possible number of labels the prediction task can have.
#       A confusion matrix of shape `(num_classes, num_classes)` will be
#       allocated to accumulate predictions from which the metric is calculated.
#     name: (Optional) string name of the metric instance.
#     dtype: (Optional) data type of the metric result.
#   Standalone usage:
#   >>> y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
#   >>> y_pred = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
#   >>>                       [0.1, 0.4, 0.5]])
#   >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
#   >>> m = metrics.OneHotMeanIoU(num_classes=3)
#   >>> m.update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
#   >>> # cm = [[0, 0, 0.2+0.4],
#   >>> #       [0.3, 0, 0],
#   >>> #       [0, 0, 0.1]]
#   >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
#   >>> # true_positives = [0, 0, 0.1]
#   >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
#   >>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
#   >>> m.result().numpy()
#   0.048
#   Usage with `compile()` API:
#   ```python
#   model.compile(
#     optimizer='sgd',
#     loss='mse',
#     metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=3)])
#   ```
#   """

#   def __init__(self, num_classes, name=None, dtype=None):
#     super(OneHotMeanIoU, self).__init__(name=name, dtype=dtype)
#     self.num_classes = num_classes 

#     self.total_cm = self.add_weight(
#         'total_confusion_matrix',
#         shape=(num_classes, num_classes),
#         initializer=init_ops.zeros_initializer,
#         dtype=dtypes.float64)

#   def update_state(self, y_true, y_pred, sample_weight=None):
#     """Accumulates the confusion matrix statistics.
#     Args:
#       y_true: The ground truth values.
#       y_pred: The predicted values.
#       sample_weight: Optional weighting of each example. Defaults to 1. Can be a
#         `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
#         be broadcastable to `y_true`.
#     Returns:
#       Update op.
#     """

#     # Select max hot-encoding channels to convert into all-class format
#     y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
#     y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

#     y_true = math_ops.cast(y_true, self._dtype)
#     y_pred = math_ops.cast(y_pred, self._dtype)

#     # Flatten the input if its rank > 1.
#     if y_pred.shape.ndims > 1:
#       y_pred = array_ops.reshape(y_pred, [-1])

#     if y_true.shape.ndims > 1:
#       y_true = array_ops.reshape(y_true, [-1])

#     if sample_weight is not None:
#       sample_weight = math_ops.cast(sample_weight, self._dtype)
#       if sample_weight.shape.ndims > 1:
#         sample_weight = array_ops.reshape(sample_weight, [-1])

#     # Accumulate the prediction to current confusion matrix.
#     current_cm = confusion_matrix.confusion_matrix(
#         y_true,
#         y_pred,
#         self.num_classes,
#         weights=sample_weight,
#         dtype=dtypes.float64)

#     return self.total_cm.assign_add(current_cm)

#   def result(self):
#     """Compute the mean intersection-over-union via the confusion matrix."""
#     sum_over_row = math_ops.cast(
#         math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
#     sum_over_col = math_ops.cast(
#         math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
#     true_positives = math_ops.cast(
#         array_ops.tensor_diag_part(self.total_cm), dtype=self._dtype)

#     # sum_over_row + sum_over_col =
#     #     2 * true_positives + false_positives + false_negatives.
#     denominator = sum_over_row + sum_over_col - true_positives

#     # The mean is only computed over classes that appear in the
#     # label or prediction tensor. If the denominator is 0, we need to
#     # ignore the class.
#     num_valid_entries = math_ops.reduce_sum(
#         math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

#     iou = math_ops.div_no_nan(true_positives, denominator)

#     return math_ops.div_no_nan(
#         math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

#   def reset_states(self):
#     K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

#   def get_config(self):
#     config = {'num_classes': self.num_classes}
#     base_config = super(OneHotMeanIoU, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))

class OneHotMeanIoU(Metric):
  """Computes mean Intersection-Over-Union metric for one-hot encoded labels.
  General definition and computation:
  Intersection-Over-Union is a common evaluation metric for semantic image
  segmentation.
  For an individual class, the IoU metric is defined as follows:
  ```
  iou = true_positives / (true_positives + false_positives + false_negatives)
  ```
  To compute IoUs, the predictions are accumulated in a confusion matrix,
  weighted by `sample_weight` and the metric is then calculated from it.
  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  This class can be used to compute the mean IoU for multi-class classification
  tasks where the labels are one-hot encoded (the last axis should have one
  dimension per class). Note that the predictions should also have the same
  shape. To compute the mean IoU, first the labels and predictions are converted
  back into integer format by taking the argmax over the class axis. Then the
  same computation steps as for the base `MeanIoU` class apply.
  Note, if there is only one channel in the labels and predictions, this class
  is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.
  Also, make sure that `num_classes` is equal to the number of classes in the
  data, to avoid a "labels out of bound" error when the confusion matrix is
  computed.
  Args:
    num_classes: The possible number of labels the prediction task can have.
      A confusion matrix of shape `(num_classes, num_classes)` will be
      allocated to accumulate predictions from which the metric is calculated.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  Standalone usage:
  >>> y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
  >>> y_pred = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
  >>>                       [0.1, 0.4, 0.5]])
  >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
  >>> m = metrics.OneHotMeanIoU(num_classes=3)
  >>> m.update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
  >>> # cm = [[0, 0, 0.2+0.4],
  >>> #       [0.3, 0, 0],
  >>> #       [0, 0, 0.1]]
  >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
  >>> # true_positives = [0, 0, 0.1]
  >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
  >>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
  >>> m.result().numpy()
  0.048
  Usage with `compile()` API:
  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.OneHotMeanIoU(num_classes=3)])
  ```
  """

  def __init__(self, num_classes, name=None, dtype=None):
    super(OneHotMeanIoU, self).__init__(name=name, dtype=dtype)
    self.num_classes = num_classes 

    self.total_cm = self.add_weight(
        'total_confusion_matrix',
        shape=(num_classes, num_classes),
        initializer=init_ops.zeros_initializer,
        dtype=dtypes.float64)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates the confusion matrix statistics.
    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    Returns:
      Update op.
    """

    # Select max hot-encoding channels to convert into all-class format
    y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)

    # Flatten the input if its rank > 1.
    if y_pred.shape.ndims > 1:
      y_pred = array_ops.reshape(y_pred, [-1])

    if y_true.shape.ndims > 1:
      y_true = array_ops.reshape(y_true, [-1])

    if sample_weight is not None:
      sample_weight = math_ops.cast(sample_weight, self._dtype)
      if sample_weight.shape.ndims > 1:
        sample_weight = array_ops.reshape(sample_weight, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        y_true,
        y_pred,
        self.num_classes+1, # +1 accounts for ignore_index class
        weights=sample_weight,
        dtype=dtypes.float64)
    
    current_cm = current_cm[:-1, :-1] 

    return self.total_cm.assign_add(current_cm)

  def result(self):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
    sum_over_col = math_ops.cast(
        math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
    true_positives = math_ops.cast(
        array_ops.tensor_diag_part(self.total_cm), dtype=self._dtype)

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = math_ops.reduce_sum(
        math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype))

    iou = math_ops.div_no_nan(true_positives, denominator)

    return math_ops.div_no_nan(
        math_ops.reduce_sum(iou, name='mean_iou'), num_valid_entries)

  def reset_states(self):
    K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

  def get_config(self):
    config = {'num_classes': self.num_classes}
    base_config = super(OneHotMeanIoU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

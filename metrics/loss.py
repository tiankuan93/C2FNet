#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library

# 3rd part packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import serialize_keras_object
import cv2
import numpy as np
from skimage import filters
# local source


def mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # y_true_weight = y_true_f / 255.0

    # loss = K.mean(K.square((y_pred - y_true) * y_true), axis=-1)
    loss = K.mean(K.square(y_true_f - y_pred_f))

    return loss


def point_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_weight = y_true_f / 255.0

    # loss = K.mean(K.square((y_pred - y_true) * y_true), axis=-1)
    # loss = K.mean(K.square(y_true_f - y_pred_f) * y_true_weight)

    sq = K.square((y_true_f - y_pred_f) * y_true_weight)
    print(sq)

    # nonzero = K.any(K.greater(y_true_f, 5.0))
    nonzero = K.greater(y_true_f, 5.0)
    n = K.sum(K.cast(nonzero, 'float32'))
    loss = K.sum(sq) / n
    return loss


def point_dis_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_weight = y_true_f / 255.0

    # loss = K.mean(K.square((y_pred - y_true) * y_true),
    #               axis=[1, 2])
    loss = K.mean(K.square(y_pred_f - y_true_f) * y_true_weight)
    return loss


def edge_dis_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f_not = (255.0 - y_true_f) / 2.0

    y_true_weight = y_true_f / 255.0

    # loss = K.mean(K.square((y_pred - y_true) * y_true),
    #               axis=[1, 2])
    loss = K.mean(K.square(y_pred_f - y_true_f_not) * y_true_weight)
    # loss = K.mean(K.square(y_pred_f - y_true_f_not) * 1.0)
    return loss


def edge_supplement_loss_v1(y_true, y_pred):
    if y_true.get_shape()[1] == 1:
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
    if y_pred.get_shape()[1] == 1:
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    y_pred = y_pred / 255.0
    sobel_edges_y_x = get_sobel_edge(y_pred)
    # sobel_edges_y_x = tf.cast(
    #     tf.greater(sobel_edges_y_x, 128.0),
    #     tf.float32)*255.0

    # bce_loss = K.binary_crossentropy(
    #     y_true*255,
    #     sobel_edges_y_x)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(sobel_edges_y_x, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_weight = y_true_f / 255.0

    # loss = K.mean(K.square((y_pred - y_true) * y_true), axis=-1)
    loss = K.mean(K.square(y_true_f - y_pred_f) * y_true_weight)

    return loss


def edge_supplement_loss(y_true, y_pred):
    print('new edge_supplement_loss mean')
    if y_true.get_shape()[1] == 1:
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
    if y_pred.get_shape()[1] == 1:
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    y_pred = y_pred / 255.0
    sobel_edges_y_x = get_sobel_edge(y_pred)
    # sobel_edges_y_x = tf.cast(
    #     tf.greater(sobel_edges_y_x, 128.0),
    #     tf.float32)*255.0

    # bce_loss = K.binary_crossentropy(
    #     y_true*255,
    #     sobel_edges_y_x)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(sobel_edges_y_x, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_weight = y_true_f / 255.0

    # loss = K.mean(K.square((y_pred - y_true) * y_true), axis=-1)
    # loss = K.mean(K.square(y_true_f - y_pred_f) * y_true_weight)

    sq = K.square((y_true_f - y_pred_f) * y_true_weight)
    print(sq)

    # nonzero = K.any(K.greater(y_true_f, 5.0))
    nonzero = K.greater(y_true_f, 5.0)
    n = K.sum(K.cast(nonzero, 'float32'))
    x_mean = K.sum(sq) / n

    return x_mean


def get_sobel_edge(img_tensor):
    sobel_edges = tf.image.sobel_edges(img_tensor)

    sobel_edges_y = sobel_edges[:, :, :, :, 0]
    sobel_edges_x = sobel_edges[:, :, :, :, 1]

    sobel_edges_y_square = tf.square(sobel_edges_y)
    sobel_edges_x_square = tf.square(sobel_edges_x)

    sobel_edges_y_x = sobel_edges_y_square + sobel_edges_x_square

    sobel_edges_y_x = sobel_edges_y_x * (255.0/tf.reduce_max(sobel_edges_y_x))
    # sobel_edges_y_x = tf.div(
    #     tf.subtract(
    #         sobel_edges_y_x,
    #         tf.reduce_min(sobel_edges_y_x)
    #     ),
    #     tf.subtract(
    #         tf.reduce_max(sobel_edges_y_x),
    #         tf.reduce_min(sobel_edges_y_x)
    #     )
    # ) * 255.0

    return sobel_edges_y_x


def edge_supplement_online_loss(y_true, y_pred):
    if y_true.get_shape()[1] == 1:
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
    if y_pred.get_shape()[1] == 1:
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    sobel_edges_y_x = get_sobel_edge(y_pred)

    # kernel = tf.ones((5, 5, 1, 1))
    # y_pred_dilate = tf.nn.conv2d(y_pred, kernel, strides=[1, 1, 1, 1],
    #                              padding='SAME')

    kernel = tf.ones((5, 5, 1))
    y_pred_dilate = tf.nn.dilation2d(
        y_pred, filter=kernel, strides=(1, 1, 1, 1),
        rates=(1, 1, 1, 1), padding="SAME")

    edge_or = tf.bitwise.bitwise_or(
            tf.cast(y_pred, tf.uint8),
            tf.cast(y_true, tf.uint8))

    edge_diate = tf.bitwise.bitwise_and(
        tf.cast(y_pred_dilate, tf.uint8),
        tf.cast(edge_or, tf.uint8))

    edge_sup = tf.cast(edge_diate, tf.float32) - y_pred

    edge_sup = tf.clip_by_value(edge_sup, 0.0, 255.0)

    edge_sup_weight = tf.cast(
        tf.greater(edge_sup, 30.0),
        tf.float32)*255.0 / 255.0

    y_true = tf.cast(edge_sup, tf.float32)
    y_pred = tf.cast(sobel_edges_y_x, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_weight = K.flatten(edge_sup_weight)

    # loss = K.mean(K.square((y_pred - y_true) * y_true), axis=-1)
    loss = K.mean(K.square(y_true_f - y_pred_f) * y_true_weight)

    return sobel_edges_y_x


def fake_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_pred_label = tf.cast(K.greater(y_pred_f, 127), tf.float32) * 255.0
    # loss = y_pred_label
    loss = K.mean(K.square(y_pred_f - y_pred_label))
    return loss


if __name__ == '__main__':
    point_dis = cv2.imread('bladder_02_point_dis.png', cv2.IMREAD_GRAYSCALE)
    edge_dis = cv2.imread('bladder_02_edge_dis.png', cv2.IMREAD_GRAYSCALE)

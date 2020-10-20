#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import time

# 3rd part packages
from keras.callbacks import Callback
import numpy as np
import warnings
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
import cv2
import tensorflow as tf
from keras import backend as K

# local source


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, best_filepath, last_filepath,
                 monitor='val_loss', verbose=0,
                 save_best=False, save_weights_only=False,
                 save_last=True,
                 mode='auto', period=1,
                 model_size=1024,
                 val_dir=''):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best_filepath = best_filepath
        self.last_filepath = last_filepath
        self.save_best = save_best
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.model_size = model_size
        self.val_model = None
        self.val_dir = val_dir

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # model = self.model.layers[-2]
        model = self.model
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            best_filepath = self.best_filepath.format(epoch=epoch + 1, **logs)
            if self.save_last:
                model.save(self.last_filepath, overwrite=True)
            if self.save_best:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with '
                                  '%s available, '
                                  'skipping.' % self.monitor,
                                  RuntimeWarning)
                elif epoch < 200:
                    return
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved '
                                  'from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, best_filepath))
                        self.best = current
                        if self.save_weights_only:
                            model.save_weights(best_filepath,
                                               overwrite=True)
                        else:
                            model.save(best_filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not '
                                  'improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1,
                                                                best_filepath))
                # if self.save_weights_only:
                #     model.save_weights(best_filepath, overwrite=True)
                # else:
                #     model.save(best_filepath, overwrite=True)

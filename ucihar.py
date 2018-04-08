#!/usr/bin/env python

'''
Loads the UCI HAR Dataset
'''

import os

import logging
log = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf
import pandas as pd

X_DIMMENSIONS = 561
Y_CLASSES = 6

def load_feature_names():
    filename = os.path.join(os.path.curdir, 'UCI-HAR-Dataset', 'features.txt')
    r = []
    with open(filename) as f:
        for line in f:
            r.append(line.strip().split(' ')[1])
    return r

# FEATURE_NAMES = load_feature_names() # duplication exists
FEATURE_NAMES = [str(i) for i in range(X_DIMMENSIONS)]

def load_data(train_or_test, x_or_y):
    '''read txt into numpy array'''
    filename = os.path.join(os.path.curdir, 'UCI-HAR-Dataset', train_or_test, "{0}_{1}.txt".format(x_or_y, train_or_test))
    
    if(x_or_y == 'y'):
        arr = np.genfromtxt(filename, dtype='int32') - 1
    else:
        arr = np.genfromtxt(filename, dtype='float32')

    log.info('loaded from txt: %s %s, %s', train_or_test, x_or_y, arr.shape)
    return arr


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(pd.DataFrame(features, columns=FEATURE_NAMES)), 
        labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(pd.DataFrame(features, columns=FEATURE_NAMES)), 
        labels))
    return dataset.batch(1)
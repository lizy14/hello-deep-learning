#!/usr/bin/env python

'''
Loads the UCI HAR Dataset
'''

import os

import logging
log = logging.getLogger(__name__)

import numpy as np

X_DIMMENSIONS = 561
Y_CLASSES = 6

def load_data(train_or_test, x_or_y):
    '''read txt into numpy array'''
    filename = os.path.join(os.path.curdir, 'UCI-HAR-Dataset', train_or_test, "{0}_{1}.txt".format(x_or_y, train_or_test))
    
    if(x_or_y == 'y'):
        arr = np.genfromtxt(filename, dtype='int32')
    else:
        arr = np.genfromtxt(filename, dtype='float32')

    log.info('loaded from txt: %s %s, %s', train_or_test, x_or_y, arr.shape)
    return arr

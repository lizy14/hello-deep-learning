#!/usr/bin/env python

'''
entry point for UCI HAR classification
'''

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

import numpy as np
import ucihar
from keras.models import Sequential
from keras.layers import Dense, Activation

def one_hot_encoding(raw, number_of_classes):
    number_of_lines = raw.shape[0]
    arr = np.zeros((number_of_lines, number_of_classes), dtype='int8')
    arr[np.arange(number_of_lines), raw - 1] = 1
    return arr

def main():
    x_train = ucihar.load_data('train', 'X')
    y_train = ucihar.load_data('train', 'y')
    y_train = one_hot_encoding(y_train, ucihar.Y_CLASSES)


    model = Sequential()
    model.add(Dense(units=100, input_dim=ucihar.X_DIMMENSIONS))
    model.add(Activation('relu'))
    model.add(Dense(units=ucihar.Y_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10, batch_size=32)


    x_test = ucihar.load_data('test', 'X')
    y_test = ucihar.load_data('test', 'y')
    y_test = one_hot_encoding(y_test, ucihar.Y_CLASSES)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

    log.info("test set loss: %s", loss_and_metrics[0])
    log.info("test set accuracy: %s", loss_and_metrics[1])

if __name__ == '__main__':
    main()
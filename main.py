#!/usr/bin/env python

'''
entry point for UCI HAR classification
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

import ucihar
import tensorflow as tf


def main():
    x_train = ucihar.load_data('train', 'X')
    y_train = ucihar.load_data('train', 'y')

    x_test = ucihar.load_data('test', 'X')
    y_test = ucihar.load_data('test', 'y')
    
    classifier = tf.estimator.DNNClassifier(
        hidden_units=[50],
        feature_columns=[tf.feature_column.numeric_column(i) for i in ucihar.FEATURE_NAMES],
        n_classes=ucihar.Y_CLASSES)
    
    classifier.train(
        input_fn=lambda: ucihar.train_input_fn(features=x_train, labels=y_train, batch_size=64), 
        steps=2000)

    eval_result = classifier.evaluate(
        input_fn=lambda: ucihar.eval_input_fn(features=x_test, labels=y_test))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    
if __name__ == '__main__':
    main()

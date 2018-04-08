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

import numpy as np
import ucihar
from neural_net import TwoLayerNet


def main():
    x_train = ucihar.load_data('train', 'X')
    y_train = ucihar.load_data('train', 'y')

    x_test = ucihar.load_data('test', 'X')
    y_test = ucihar.load_data('test', 'y')

    net = TwoLayerNet(
        input_size=ucihar.X_DIMMENSIONS,
        hidden_size=50,
        output_size=ucihar.Y_CLASSES)

    stats = net.train(
        x_train, y_train, x_test, y_test,
        num_iters=100000, batch_size=64,
        learning_rate=1e-2, learning_rate_decay=1.)

    predictions = net.predict(x_test)
    val_acc = (predictions == y_test).mean()
    print('Validation accuracy: ', val_acc)
    
    try: 
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test, predictions)) 
        print(classification_report(y_test, predictions)) 
    except ImportError: 
        pass

    try:
        import matplotlib.pyplot as plt
        # Plot the loss function and train / validation accuracies
        plt.subplot(1, 2, 1)
        plt.plot(stats['loss_history'])
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        train_plot, = plt.plot(stats['train_acc_history'], label='train')
        val_plot, = plt.plot(stats['val_acc_history'], label='val')
        plt.legend(handles=[train_plot, val_plot])
        plt.title('Classification accuracy history')
        plt.xlabel('Epoch')
        plt.ylabel('Clasification accuracy')
        plt.show()
    except ImportError:
        pass


if __name__ == '__main__':
    main()

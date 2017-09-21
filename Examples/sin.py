import theano
theano.config.optimizer='None'

import numpy as np
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
import lasagne.updates
import underoverfitting
import random


def create_net_shape(**kwargs):
    from nolearn.lasagne import BatchIterator
    args = {'layers': [('input', layers.InputLayer),
                       ('dense1', layers.DenseLayer),
                       ('dropout1', layers.DropoutLayer),
                       ('dense2', layers.DenseLayer),
                       ('dropout2', layers.DropoutLayer),
                       ('output', layers.DenseLayer),
                       ],
            # dense1
            'dense1_num_units': 100,
            'dense1_nonlinearity': lasagne.nonlinearities.rectify,
            'dense1_num_leading_axes': 1,
            # dropout1
            'dropout1_p': 0.5,
            'dropout1_shared_axes': (0,),
            # dense2
            'dense2_num_units': 100,
            'dense2_nonlinearity': lasagne.nonlinearities.rectify,
            'dense2_num_leading_axes': 1,
            # dropout2
            'dropout2_p': 0.5,
            'dropout2_shared_axes': (0,),
            # output
            'output_nonlinearity': lasagne.nonlinearities.softmax,
            'output_num_leading_axes': 1,
            # optimization method params
            'update_learning_rate': 0.01,
            'update_momentum': 0.9,
            'max_epochs': 50,
            'verbose': 1}

    # override default arguments with specified arguments
    for key in kwargs.keys():
        args[key] = kwargs[key]

    nn = NeuralNet(**args)
    nn.initialize()  # TODO: probably shouldn't be here
    return nn


def do():
    random.seed(10)
    n = 1000
    percentage_dev = 0.1

    train_data_1d = np.random.uniform(0, 1, int((1-percentage_dev) * n)).astype(np.float32)
    dev_data_1d = np.random.uniform(0, 1, int(percentage_dev * n)).astype(np.float32)
    train_data = np.array([[x, x] for x in train_data_1d]).astype(np.float32)
    dev_data = np.array([[x, x] for x in dev_data_1d]).astype(np.float32)

    train_truths = np.array([truth(x) for x in train_data]).astype(np.int32)
    dev_truths = np.array([truth(x) for x in dev_data]).astype(np.int32)

    print('data shape {}'.format(train_data.shape))
    print('truth shape {}'.format(train_truths.shape))
    underoverfitting.train_and_predict_and_plot(train_data, train_truths, dev_data, dev_truths, create_net_shape)


def truth(x):
    return int(x[0] * 10)


if __name__ == '__main__':
    do()


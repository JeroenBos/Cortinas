import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
import lasagne.updates
from Examples.General1D import train


def create_net_shape(**kwargs):
    args = {'layers': [('input', layers.InputLayer),
                       ('dense1', layers.DenseLayer),
                       ('dropout1', layers.DropoutLayer),
                       ('dense2', layers.DenseLayer),
                       ('dropout2', layers.DropoutLayer),
                       ('output', layers.DenseLayer),
                       ],
            # dense1
            'dense1_num_units': 1000,
            'dense1_nonlinearity': lasagne.nonlinearities.rectify,
            'dense1_num_leading_axes': 1,
            # dropout1
            'dropout1_p': 0.5,
            'dropout1_shared_axes': (0,),
            # dense2
            'dense2_num_units': 1000,
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
    nn.initialize()
    return nn


def truth(x):
    return int(x[0] * 10)


if __name__ == '__main__':
    train(truth, create_net_shape)

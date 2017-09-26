import math
from lasagne import layers
import lasagne.updates
from Examples.General1D import minimize
from HyperparameterDistributions.NaturalNumberDistribution import NaturalNumberDistribution
from HyperparameterDimension import HyperparameterDimension

CLASS_SIZE = 5

default_args = {'layers': [('input', layers.InputLayer),
                           ('dense1', layers.DenseLayer),
                           ('dropout1', layers.DropoutLayer),
                           ('dense2', layers.DenseLayer),
                           ('dropout2', layers.DropoutLayer),
                           ('output', layers.DenseLayer),
                           ],
                # input
                'input_shape': (None, 2),
                # dense1
                # parameterized: 'dense1_num_units': 1000,
                'dense1_nonlinearity': lasagne.nonlinearities.rectify,
                'dense1_num_leading_axes': 1,
                # dropout1
                'dropout1_p': 0.5,
                'dropout1_shared_axes': (0,),
                # dense2
                # parameterized: 'dense2_num_units': 1000,
                'dense2_nonlinearity': lasagne.nonlinearities.rectify,
                'dense2_num_leading_axes': 1,
                # dropout2
                'dropout2_p': 0.5,
                'dropout2_shared_axes': (0,),
                # output
                'output_nonlinearity': lasagne.nonlinearities.softmax,
                'output_num_leading_axes': 1,
                # optimization method params
                'update_learning_rate': 0.1,
                'output_num_units': CLASS_SIZE,
                'update_momentum': 0.9,
                'max_epochs': 50,
                'verbose': 1}


def truth(x):
    x_denormalized = x * 10
    result = math.sin(x_denormalized)
    result_classified = int((result + 1) / 2 * CLASS_SIZE)
    return result_classified


if __name__ == '__main__':
    minimize(truth, default_args, [HyperparameterDimension('dense1_num_units', NaturalNumberDistribution()),
                                   HyperparameterDimension('dense2_num_units', NaturalNumberDistribution())])

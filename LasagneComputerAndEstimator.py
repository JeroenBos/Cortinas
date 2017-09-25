from nolearn.lasagne import NeuralNet
from ComputerAndEstimator import ComputerAndEstimator
import nn_size
import copy


class LasagneComputerAndEstimator(ComputerAndEstimator):

    def __init__(self, default_nn_init_args):
        self._cache = {}
        self.__default_nn_init_args = default_nn_init_args
        super(self.get_or_create_nn, nn_size.compute_size_for(self.get_or_create_nn))

    def get_or_create_nn(self, v):
        try:
            result = self._cache[v]
        except KeyError:
            result = self._create_nn(v)
            self._cache[v] = result
        return result

    def _create_nn(self, overriding_args):
        args = copy.copy(self.__default_nn_init_args)
        return create_nn(args, overriding_args)


def create_nn(args, overriding_args):
    for key in overriding_args.keys():
        args[key] = overriding_args[key]
    nn = NeuralNet(**args)
    # nn.initialize()
    return nn




from nolearn.lasagne import NeuralNet
from ComputerAndEstimator import ComputerAndEstimator
import nn_size
import copy
from Estimation.OneDimensionalEstimator import OneDimensionalEstimator
from Estimation.ParabolicEstimationTechnique import ParabolicEstimationTechnique


class LasagneComputerAndEstimator(ComputerAndEstimator):

    def __init__(self, default_nn_init_args, evaluate):
        self._cache = {}
        self.__compute = evaluate
        self.__default_nn_init_args = default_nn_init_args

        def construct_estimator(get_cached_error):
            def error_selection(get_cached_error_):
                def get_cache(v):
                    result = get_cached_error_(v)
                    return result[0] if result is not None else None
                return get_cache
            return OneDimensionalEstimator(ParabolicEstimationTechnique(), error_selection(get_cached_error))
        super().__init__(self._compute, construct_estimator)

    def _compute(self, v):
        nn = self.get_or_create_nn(v)
        return self.__compute(nn)

    def get_or_create_nn(self, v):
        try:
            result = self._cache[v]
        except KeyError:
            result = self._create_nn(v)
            self._cache[v] = result
        assert result is not None
        return result

    def _create_nn(self, overriding_args):
        args = copy.copy(self.__default_nn_init_args)
        return create_nn(args, overriding_args)

    def try_compute_cost(self, v):
        return nn_size.compute_size_for(self.get_or_create_nn(v))


def create_nn(args, overriding_args):
    for key in overriding_args.keys():
        args[key] = overriding_args[key]
    nn = NeuralNet(**args)
    nn.initialize()
    return nn


def lasagne_weigh(error, _cost, _v):
    return error





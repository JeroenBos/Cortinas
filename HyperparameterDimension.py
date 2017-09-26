from HyperparameterDistributions.RealNumberDistribution import RealNumberDistribution


class HyperparameterDimension:
    @property
    def key(self):
        return self.__key

    @property
    def distribution(self):
        return self.__distribution

    def __init__(self, key, distribution=None):
        assert not isinstance(key, int)
        self.__key = key
        self.__distribution = distribution if distribution is not None else RealNumberDistribution()

    def step(self, current, step_size):
        index = self.__distribution.index(current)
        return self.__distribution.get(index + step_size, None)

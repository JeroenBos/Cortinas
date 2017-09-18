import copy
import HyperparameterDimension


class Hyperparameter:

    @property
    def dimensions(self):
        return self.__dimensions

    def __init__(self, dimensions, values):
        assert len(dimensions) == len(values)
        self.__dimensions = dimensions
        self.__values = values
        self.__hash = sum(values)

    def _dimension_index(self, key):
        for index, dimension in enumerate(self.dimensions):
            if dimension.key == key:
                return index
        else:
            return None

    def __iter__(self):
        return (self[index] for index in range(0, len(self.__dimensions)))

    def __getitem__(self, key):
        key = key.key if isinstance(key, HyperparameterDimension.HyperparameterDimension) else key
        index = key if isinstance(key, int) else self._dimension_index(key)
        assert index is not None, '{} is not a valid key'.format(key)
        assert 0 <= index < len(self.__values)

        return self.__values[index]

    def with_(self, key, value):
        """
        Creates a new hyperparameter with the specified value. So this behavior is asymmetric to __getitem__
        :return: the new hyperparameter
        """
        assert value is not None

        values = copy.copy(self.__values)
        values[self._dimension_index(key)] = value
        return Hyperparameter(self.dimensions, values)

    def step(self, dimension, step_size):
        assert isinstance(dimension, HyperparameterDimension.HyperparameterDimension)
        x = dimension.step(self[dimension.key], step_size)
        return self.with_(dimension.key, x) if x is not None else None

    def __eq__(self, other):
        if other is self:
            return True
        if isinstance(other, Hyperparameter):
            return self.__eq__(other.__values)
        if not hasattr(other, '__len__'):
            return self.__eq__([other])
        elif len(self.__values) == len(other):
            for i in range(0, len(self.__values)):
                if self.__values[i] != other[i]:
                    return False
            return True
        return False

    def __hash__(self):
        return self.__hash

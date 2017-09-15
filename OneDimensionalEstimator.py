class OneDimensionalEstimator:  # selects one dimension to use in estimating

    def __init__(self, technique_1d, cached_error):
        """
        :param technique_1d: A function taking a list of (float, TError)-tuples and a float to estimate the error at
        :param cached_error: A function taking a vector and retrieving a TError; or None if it hasn't been computed yet
        """
        self.__technique = technique_1d
        self.__cached_error = cached_error

    @staticmethod
    def tuple_with(v, x, dimension):
        new_v = list(v)
        new_v[dimension] = x
        r = tuple(new_v)
        return r

    def estimate(self, v, dimension, direction):
        """
    Estimates L at x
        :param v:
        :param dimension: the direction of dx that resulted in x
        :param direction:
        :return:
        """
        assert direction in [-1, 1]

        def to_v(x):
            return self.tuple_with(v, x, dimension)

        def get_cached_error_of_x(x):
            return self.__cached_error(to_v(x))

        return self._estimate1d(v[dimension], dimension, direction, get_cached_error_of_x)

    def _estimate1d(self, x, dimension_of_x, direction, get_cached_error_of_x):
        print(get_cached_error_of_x)
        if direction == 1:
            d1 = -2 * direction
            d2 = -direction
            d3 = direction
        else:
            d1 = direction
            d2 = -direction
            d3 = 2 * direction

        x_coordinates = [self._step(x, dimension_of_x, d) for d in [d1, d2, d3]]
        points = [(x, get_cached_error_of_x(x)) for x in x_coordinates if get_cached_error_of_x(x) is not None]
        return self.__technique.estimate(points, x)

    @staticmethod
    def _step(x, _dimension_of_x, step):   # TODO: replace by some call to dimension_of_x
        return x + step

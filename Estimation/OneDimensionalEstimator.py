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

    def estimate(self, v, dx):
        """
    Estimates L at v
        :param v:
        :param dx: a tuple of the dimension in v that changes, and the offset that resulted in v
        :return:
        """
        assert v is not None
        dimension, offset = dx
        assert offset in [-1, 1]

        def to_v(x):
            return v.with_(dimension.key, x)

        def get_cached_error_of_x(x):
            return self.__cached_error(to_v(x))

        return self._estimate1d(v[dimension], dimension, offset, get_cached_error_of_x)

    def _estimate1d(self, x, dimension_of_x, direction, get_cached_error_of_x):
        if direction == 1:
            d1 = -2 * direction
            d2 = -direction
            d3 = direction
        else:
            d1 = direction
            d2 = -direction
            d3 = 2 * direction

        x_coordinates = [dimension_of_x.step(x, d) for d in [d1, d2, d3]]
        points = []
        for x in x_coordinates:
            if x is not None:
                if get_cached_error_of_x(x) is not None:
                    points.append((x, get_cached_error_of_x(x)))
        return self.__technique.estimate(points, x)

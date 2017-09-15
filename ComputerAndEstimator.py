class ComputerAndEstimator:

    def __init__(self, compute, get_estimator):
        """
        :param compute: A function that takes a vector and computes an error
        :param get_estimator: A function that takes a function that takes a vector and returns an optional error
                                    and returns an error estimator.
        """
        self.__cache = {}
        self.estimate = get_estimator(self._get_cached).estimate
        print(type(self.estimate))
        self.__compute = compute

    def _get_cached(self, v):
        return self.__cache.get(v, None)

    def compute(self, v):
        result = self.__cache.get(v, None)
        if result is None:
            result = self.__compute(v)
            self.__cache[v] = result
        return result


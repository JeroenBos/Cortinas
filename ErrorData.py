class TestErrorData:

    @property
    def train_bias(self):
        return self.__train_bias

    @property
    def dev_bias(self):
        return self.__dev_bias

    def __init__(self, train_set_bias: float, dev_set_bias: float):
        self.__train_bias = train_set_bias
        self.__dev_bias = dev_set_bias



    def __lt__(self, other):
        return self.__magnitude < other.__magnitude

    def __float__(self):
        return self.__magnitude

    def __repr__(self):
        return str(self.__magnitude)

    @staticmethod
    def estimate(c1, c2, c3, v, dimension):
        return greedydescent.fit_estimator((c1[0], c1[1].__magnitude if c1[1] is not None else None),
                                           (c2[0], c2[1].__magnitude if c2[1] is not None else None),
                                           (c3[0], c3[1].__magnitude if c3[1] is not None else None),
                                           v,
                                           dimension)


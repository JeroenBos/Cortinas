class ErrorData:

    def __init__(self, train_set_bias: float, dev_set_bias: float):
        self.__train_bias = train_set_bias
        self.__dev_bias = dev_set_bias

    @property
    def train_bias(self):
        return self.__train_bias

    @property
    def dev_bias(self):
        return self.__dev_bias

    def __lt__(self, other):
        return float(self) < float(other)

    def __float__(self):
        return 0.5 * (self.train_bias - self.dev_bias) ** 2 \
               + self.train_bias ** 2 \
               + self.dev_bias ** 2

    def __repr__(self):
        def format_digits(n):
            return str(round(n, 2))
        return '(train bias = %s, dev bias = %s)'.format(format_digits(self.train_bias), format_digits(self.dev_bias))

    def weigh(self, cost, _v):
        """
        Weighs the computational expense of NN(v) against the expected performance of NN(v)
        :param cost: A number indicating the computational expense of NN(v)
        :param _v: The hyperparameter defining a NN
        :return: A totally ordered comparable token signifying the weighed result of NN(v)
        i.e.
        """
        return ErrorData.ErrorCostData(self, cost)

    class ErrorCostData:

        def __init__(self, error_data, cost):
            self.__error = error_data
            self.__cost = cost

        def __lt__(self, other):
            return float(self) < float(other)

        def __float__(self):
            return 100000 * self.__error - self.__cost



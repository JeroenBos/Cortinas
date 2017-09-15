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
        return 0.5 * (self.train_bias - self.dev_bias) ** 2\
               + self.train_bias ** 2\
               + self.dev_bias ** 2

    def __repr__(self):
        def format_digits(x):
            return str(round(x, 2))
        return '(train bias = %s, dev bias = %s)'.format(format_digits(self.train_bias), format_digits(self.dev_bias))

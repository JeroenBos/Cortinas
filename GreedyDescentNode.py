class GreedyDescentNode:

    def __init__(self, x, error):
        assert error is not None
        self.__error = error
        self.__x = x

    @property
    def error(self):
        return self.__error

    @property
    def x(self):
        return self.__x

    def __lt__(self, other):
        return self.error < other.error

    def __eq__(self, other):
        if other is GreedyDescentNode:
            return other.__x == self.__x
        elif isinstance(other, type(self.__x)):
            return other == self.__x
        else:
            return False

    def __repr__(self):
        return str(self.x) + ", cost = " + str(self.error)

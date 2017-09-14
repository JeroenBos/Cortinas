class GreedyDescentNode:

    def __init__(self, x, cost):
        self.__cost = cost
        self.__x = x

    @property
    def cost(self):
        return self.__cost

    @property
    def x(self):
        return self.__x

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        if other is GreedyDescentNode:
            return other.__x == self.__x
        elif isinstance(other, type(self.__x)):
            return other == self.__x
        else:
            return False


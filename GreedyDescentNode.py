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
        return self.__x == other.__x


class CollectionDistribution:

    def __init(self, collection):
        self.__collection = collection

    def index(self, item):
        return self.__collection.index(item)

    def get(self, index, _default):
        assert isinstance(index, int)
        return self.__collection[index] if 0 <= index < len(self.__collection) else None

    def default(self):
        return self.__collection[len(self.__collection) / 2]

class RealNumberDistribution:

    @staticmethod
    def index(item):
        assert isinstance(item, int)
        return item

    @staticmethod
    def get(index, _default):
        assert isinstance(index, int)
        return index

    @property
    def default(self):
        return 0

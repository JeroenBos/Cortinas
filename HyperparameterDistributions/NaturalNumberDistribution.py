class NaturalNumberDistribution:

    @staticmethod
    def index(item):
        assert isinstance(item, int)
        return item if item > 0 else None

    @staticmethod
    def get(index, default):
        assert isinstance(index, int)
        return index if index > 0 else default
class NaturalNumberDistribution:

    def index(self, item):
        assert isinstance(item, int)

        if item < 0:
            return None

        steps_from_default, rem = divmod(item - self.default, self.__step)
        if rem != 0:
            return None  # the item is 'stepped over'

        offset = self.__default // self.__step
        return steps_from_default + offset

    def get(self, index, default):
        assert isinstance(index, int)

        rough_result = index // self.__step
        default_offset = self.__default % self.__step
        result = rough_result + default_offset
        return result if result >= 0 else default

    @property
    def default(self):
        return self.__default

    def __init__(self, default=30, step=1):
        assert default >= 0
        assert step > 0
        self.__default = default
        self.__step = step

class ParabolicEstimationTechnique:

    @staticmethod
    def estimate(data, x):
        if data is None or len(data) == 0:
            return None
        if len(data) == 1:
            return data[0][1]
        if len(data) == 2:
            return ParabolicEstimationTechnique.estimate2(data[0], data[1], x)
        if len(data) == 3:
            return ParabolicEstimationTechnique.estimate3(data[0], data[1], data[2], x)
        raise NotImplementedError('The length of data is too long')

    @staticmethod
    def are_unique(elements):
        seen = set()
        return not any(i in seen or seen.add(i) for i in elements)

    @staticmethod
    def estimate3(coordinate1, coordinate2, coordinate3, x):
        """
Fits a parabola to the specified coordinates in the specified dimension and estimates the y value at v
    :param coordinate1:
    :param coordinate2:
    :param coordinate3:
    :param x:
    :return:
    """
        x1, y1 = coordinate1
        x2, y2 = coordinate2
        x3, y3 = coordinate3
        assert None not in [y1, y2, y3]
        assert ParabolicEstimationTechnique.are_unique([x1, x2, x3, x]), 'The specified x values must differ'

        d = (x1 - x2) * (x1 - x3) * (x2 - x3)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d
        c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / d
        return a * x * x + b * x + c

    @staticmethod
    def estimate2(coordinate1, coordinate2, x):
        """
Fits a line to the specified coordinates in the specified dimension and estimates the y value at v
    :param coordinate1:
    :param coordinate2:
    :param x:
    :return:
    """
        x1, error1 = coordinate1
        x2, error2 = coordinate2

        assert error1 is not None
        assert error2 is not None
        assert x1 is not x2, 'The specified x values must differ'

        a = (error2 - error1) / (x2 - x1)
        b = error1 - a * x1
        return a * x + b

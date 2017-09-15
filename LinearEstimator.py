class LinearEstimator:

    @staticmethod
    def are_unique(elements):
        seen = set()
        return not any(i in seen or seen.add(i) for i in elements)

    @staticmethod
    def estimate3(coordinate1, coordinate2, coordinate3, v, dimension):
        """
Fits a parabola to the specified coordinates in the specified dimension and estimates the y value at v
    :param coordinate1:
    :param coordinate2:
    :param coordinate3:
    :param v:
    :param dimension:
    :return:
    """
        v1, y1 = coordinate1
        v2, y2 = coordinate2
        v3, y3 = coordinate3
        x1, x2, x3, x = v1[dimension], v2[dimension], v3[dimension], v[dimension]
        assert LinearEstimator.are_unique([x1, x2, x3, x]), ValueError('The specified x values must differ')

        d = (x1 - x2) * (x1 - x3) * (x2 - x3)
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d
        c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / d
        return a * x * x + b * x + c

    @staticmethod
    def estimate2(coordinate1, coordinate2, v, dimension):
        """
Fits a line to the specified coordinates in the specified dimension and estimates the y value at v
    :param coordinate1:
    :param coordinate2:
    :param v:
    :param dimension:
    :return:
    """
        v1, error1 = coordinate1
        v2, error2 = coordinate2

        assert error1 is not None
        assert error2 is not None
        assert v1 is not v2, 'The specified x values must differ'

        x1, x2, x = v1[dimension], v2[dimension], v[dimension]

        a = (error2 - error1) / (x2 - x1)
        b = error1 - a * x1
        return a * x + b

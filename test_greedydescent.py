import unittest
import greedydescent
from GreedyDescentNode import GreedyDescentNode
from LinearEstimator import LinearEstimator


class TestLinearEstimator(unittest.TestCase):

    def test_simple(self):
        y = greedydescent.fit_estimator2(((1,), 1), ((2,), 2), (3,), 0, LinearEstimator)
        self.assertEqual(y, 3)

    def test_simple_floats(self):
        y = greedydescent.fit_estimator2(((1.0,), 1.0), ((2.0,), 2.0), (4.0,), 0, LinearEstimator)
        self.assertEqual(y, 4.0)

    # def test_xs_must_differ(self):
    #     with self.assertRaises(ValueError):
    #         greedydescent.fit_estimator2((1.0, 1.0), (1.0, 1.0), 4.0, 0, LinearEstimator)


class TestParabolaEstimator(unittest.TestCase):

    def test_simple(self):
        y = greedydescent.fit_estimator(((0,), 0), ((1,), 1), ((2,), 4), (3,), 0, LinearEstimator)
        self.assertEqual(y, 9)

    def test_not_so_simple(self):  # 6 x^2 - 3 x - 10
        y = greedydescent.fit_estimator(((-2,), 20), ((1,), -7), ((2,), 8), (0.5,), 0, LinearEstimator)
        self.assertEqual(y, -10.0)

    def test_deferral_to_linear(self):
        y = greedydescent.fit_estimator(((1,), None), ((1,), 1), ((2,), 2), (3,), 0, LinearEstimator)
        self.assertEqual(y, 3)


class ComputeNeighboringXs(unittest.TestCase):

    def test_simple(self):
        next_x = greedydescent.compute_next_x((0, 0), 1, 1)
        self.assertEqual(next_x, (0, 1))

    def test_negative(self):
        next_x = greedydescent.compute_next_x((0, 0), 1, -1)
        self.assertEqual(next_x, (0, -1))


class TestGreedyDescentNode(unittest.TestCase):

    def test_ordering(self):
        nodes = [GreedyDescentNode(None, 2),
                 GreedyDescentNode(None, 3),
                 GreedyDescentNode(None, 1),
                 GreedyDescentNode(None, 0)]
        nodes.sort()
        self.assertEqual(nodes[0].cost, 0)
        self.assertEqual(nodes[1].cost, 1)
        self.assertEqual(nodes[2].cost, 2)
        self.assertEqual(nodes[3].cost, 3)

    def test_containment(self):
        x0 = [0]
        x1 = [1]
        nodes = [GreedyDescentNode(x0, 0),
                 GreedyDescentNode(x1, 1)]
        self.assertTrue(x0 in nodes)


class TestGreedyDescent(unittest.TestCase):

    def test_parabola(self):
        f_cache = {}

        def f(x):
            value = x[0]
            result = value * value - 10 * value + 6
            f_cache[x] = result
            return result

        seed = (-20,)

        def j(x, must_compute: bool):
            if must_compute:
                return f(x)
            else:
                return f_cache.get(x, None)

        def cost_heuristic(x):
            return 0

        def weigh_cost_loss(estimated_loss, estimated_cost, x):
            return estimated_cost

        def debug(arg):
            print(arg)

        def abort(_, __, consecutive_higher):
            return consecutive_higher > 10

        for result in sorted(greedydescent.minimize(j,
                                                    [seed],
                                                    cost_heuristic,
                                                    weigh_cost_loss,
                                                    LinearEstimator,
                                                    abort=abort,
                                                    debug=debug)):
            print('x = ' + str(result.x) + ' with cost ' + str(result.cost))

    def test_hyper_parabola(self):

        f_cache = {}

        def f(coordinate):
            x, y = coordinate
            result_ = x * x - 10 * x + 6 * y + x * y + y * y

            nonlocal f_cache
            f_cache[coordinate] = result_
            return result_

        seed = (10, 10)

        def j(x, must_compute: bool):
            if must_compute:
                return f(x)
            else:
                nonlocal f_cache
                result_ = f_cache.get(x, None)
                return result_

        def cost_heuristic(x):
            return 0

        def weigh_cost_loss(estimated_loss, estimated_cost, x):
            if estimated_loss is not None:
                return estimated_loss
            return (estimated_loss if estimated_loss is not None else 0) - estimated_cost

        def debug(arg):
            print('{' + str(arg[0]) + ', ' + str(arg[1]) + '},')

        def abort(_, __, consecutive_higher):
            return consecutive_higher > 20

        for result in sorted(greedydescent.minimize(j,
                                                    [seed],
                                                    cost_heuristic,
                                                    weigh_cost_loss,
                                                    LinearEstimator,
                                                    abort=abort,
                                                    debug=debug)):
            print('x = ' + str(result.x) + ' with cost ' + str(result.cost))

    def test_error_type_parameter(self):

        f_cache = {}

        def f(coordinate):
            x, y = coordinate
            result_ = TestErrorData(x * x - 10 * x + 6 * y + x * y + y * y)

            nonlocal f_cache
            f_cache[coordinate] = result_
            return result_

        seed = (10, 10)

        def j(x, must_compute: bool):
            if must_compute:
                return f(x)
            else:
                nonlocal f_cache
                result_ = f_cache.get(x, None)
                return result_

        def cost_heuristic(x):
            return 0

        def weigh_cost_loss(estimated_loss, estimated_cost, x):
            if estimated_loss is not None:
                return estimated_loss
            return estimated_loss if estimated_loss is not None else 0

        def debug(arg):
            print('{' + str(arg[0]) + ', ' + str(arg[1]) + '},')

        def abort(_, __, consecutive_higher):
            return consecutive_higher > 20

        for result in sorted(greedydescent.minimize(j,
                                                    [seed],
                                                    cost_heuristic,
                                                    weigh_cost_loss,
                                                    TestErrorData,
                                                    abort=abort,
                                                    debug=debug)):
            print('x = ' + str(result.x) + ' with cost ' + str(result.cost))


class TestErrorData:

    def __init__(self, magnitude):
        self.__magnitude = magnitude

    def __lt__(self, other):
        return self.__magnitude < other.__magnitude

    def __repr__(self):
        return str(self.__magnitude)

    @staticmethod
    def estimate3(c1, c2, c3, v, dimension):
        return greedydescent.fit_estimator((c1[0], c1[1].__magnitude if c1[1] is not None else None),
                                           (c2[0], c2[1].__magnitude if c2[1] is not None else None),
                                           (c3[0], c3[1].__magnitude if c3[1] is not None else None),
                                           v,
                                           dimension,
                                           LinearEstimator)

    @staticmethod
    def estimate2(c1, c2, v, dimension):
        return greedydescent.fit_estimator2((c1[0], c1[1].__magnitude if c1[1] is not None else None),
                                            (c2[0], c2[1].__magnitude if c2[1] is not None else None),
                                            v,
                                            dimension,
                                            LinearEstimator)


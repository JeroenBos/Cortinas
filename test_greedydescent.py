import unittest
import greedydescent
from GreedyDescentNode import GreedyDescentNode
from Estimation.OneDimensionalEstimator import OneDimensionalEstimator
from Estimation.ParabolicEstimationTechnique import ParabolicEstimationTechnique
from ComputerAndEstimator import ComputerAndEstimator
from HyperparameterDistributions.NaturalNumberDistribution import NaturalNumberDistribution
from HyperparameterDistributions.RealNumberDistribution import RealNumberDistribution
from HyperparameterDistributions.CollectionDistribution import CollectionDistribution
from HyperparameterDimension import HyperparameterDimension
from Hyperparameter import Hyperparameter



class TestLinearEstimator(unittest.TestCase):

    def test_simple(self):
        y = ParabolicEstimationTechnique.estimate([(1, 1), (2, 2)], 3)
        self.assertEqual(y, 3)

    def test_simple_floats(self):
        y = ParabolicEstimationTechnique.estimate([(1.0, 1.0), (2.0, 2.0)], 4.0)
        self.assertEqual(y, 4.0)


class TestParabolaEstimator(unittest.TestCase):

    def test_simple(self):
        y = ParabolicEstimationTechnique.estimate([(0, 0), (1, 1), (2, 4)], 3)
        self.assertEqual(y, 9)

    def test_not_so_simple(self):  # 6 x^2 - 3 x - 10
        y = ParabolicEstimationTechnique.estimate([(-2, 20), (1, -7), (2, 8)], 0.5)
        self.assertEqual(y, -10.0)


class TestGreedyDescentNode(unittest.TestCase):

    def test_ordering(self):
        nodes = [GreedyDescentNode(None, 2),
                 GreedyDescentNode(None, 3),
                 GreedyDescentNode(None, 1),
                 GreedyDescentNode(None, 0)]
        nodes.sort()
        self.assertEqual(nodes[0].error, 0)
        self.assertEqual(nodes[1].error, 1)
        self.assertEqual(nodes[2].error, 2)
        self.assertEqual(nodes[3].error, 3)

    def test_containment(self):
        x0 = [0]
        x1 = [1]
        nodes = [GreedyDescentNode(x0, 0),
                 GreedyDescentNode(x1, 1)]
        self.assertTrue(x0 in nodes)


class TestGreedyDescent(unittest.TestCase):

    def test_parabola(self):

        dimensions = [HyperparameterDimension(key='0', distribution=RealNumberDistribution())]

        def f(x):
            value = x[0]
            result_ = value * value - 10 * value + 6
            return result_

        seed = Hyperparameter(dimensions, [-20])

        def cost_heuristic(x):
            return 0

        def weigh_cost_loss(_estimated_loss, estimated_cost, x):
            return estimated_cost

        def debug(arg):
            print(arg)

        def abort(_, __, consecutive_higher):
            return consecutive_higher > 10

        def get_technique(get_cached):
            return OneDimensionalEstimator(ParabolicEstimationTechnique, get_cached)

        result = sorted(greedydescent.minimize(ComputerAndEstimator(f, get_technique),
                                               [seed],
                                               cost_heuristic,
                                               abort=abort,
                                               debug=debug,
                                               weigh=weigh_cost_loss))
        self.assertEqual(result[0].x, (5,))
        for r in result:
            print('x = ' + str(r.x) + ' with cost ' + str(r.error))

    def test_hyper_parabola(self):

        dimensions = [HyperparameterDimension(key='0', distribution=RealNumberDistribution()),
                      HyperparameterDimension(key='1', distribution=RealNumberDistribution())]

        def f(coordinate):
            x, y = coordinate
            result_ = x * x - 10 * x + 6 * y + x * y + y * y
            return result_

        seed = Hyperparameter(dimensions, [10, 10])

        def cost_heuristic(_x):
            return 0

        def weigh_cost_loss(estimated_loss, estimated_cost, x):
            if estimated_loss is not None:
                return estimated_loss
            return (estimated_loss if estimated_loss is not None else 0) - estimated_cost

        def debug(_arg):
            pass  # print('{' + str(arg[0]) + ', ' + str(arg[1]) + '},')

        def abort(_, __, consecutive_higher):
            return consecutive_higher > 20

        def get_technique(get_cached):
            return OneDimensionalEstimator(ParabolicEstimationTechnique, get_cached)

        result = sorted(greedydescent.minimize(ComputerAndEstimator(f, get_technique),
                                               [seed],
                                               cost_heuristic,
                                               abort=abort,
                                               debug=debug,
                                               weigh=weigh_cost_loss))
        self.assertEqual(result[0].x, (8, -7))
        for r in result:
            print('x = ' + str(r.x) + ' with cost ' + str(r.error))

    def test_error_type_parameter(self):

        dimensions = [HyperparameterDimension(key='0', distribution=NaturalNumberDistribution()),
                      HyperparameterDimension(key='1', distribution=NaturalNumberDistribution())]

        def f(coordinate):
            x, y = coordinate
            result_ = TestErrorData(x * x - 10 * x + 6 * y + x * y + y * y)
            return result_

        seed = Hyperparameter(dimensions, [10, 10])

        def cost_heuristic(x):
            return 0

        def debug(arg):
            pass  # print('{' + str(arg[0]) + ', ' + str(arg[1]) + '},')

        def abort(_, __, consecutive_higher):
            return consecutive_higher > 20

        def get_technique(get_cached):
            return OneDimensionalEstimator(ParabolicEstimationTechnique, get_cached)

        for result in sorted(greedydescent.minimize(ComputerAndEstimator(f, get_technique),
                                                    [seed],
                                                    cost_heuristic,
                                                    abort=abort,
                                                    debug=debug)):
            print('x = ' + str(result.x) + ' with cost ' + str(result.error))


class TestErrorData:

    def __init__(self, magnitude):
        self.__magnitude = magnitude

    def __lt__(self, other):
        return self.__magnitude < other.__magnitude

    def __float__(self):
        return float(self.__magnitude)

    def __repr__(self):
        return str(self.__magnitude)

    def weigh(self, _cost, _v):
        return self if self is not None else 0

    def __add__(self, other):
        return TestErrorData(self.__magnitude + other.__magnitude)

    def __sub__(self, other):
        return TestErrorData(self.__magnitude - other.__magnitude)

    def __mul__(self, other):
        return TestErrorData(self.__magnitude * other)

    def __truediv__(self, other):
        return TestErrorData(self.__magnitude / other)

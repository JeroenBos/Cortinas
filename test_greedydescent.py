import unittest
import greedydescent
from GreedyDescentNode import GreedyDescentNode


class TestLinearEstimator(unittest.TestCase):

    def test_simple(self):
        y = greedydescent.linear_fit_estimator(1, 1, 2, 2, 3)
        self.assertEqual(y, 3)

    def test_simple_floats(self):
        y = greedydescent.linear_fit_estimator(1.0, 1.0, 2.0, 2.0, 4.0)
        self.assertEqual(y, 4.0)

    def test_xs_must_differ(self):
        with self.assertRaises(ValueError):
            greedydescent.linear_fit_estimator(1.0, 1.0, 1.0, 1.0, 4.0)


class TestParabolaEstimator(unittest.TestCase):

    def test_simple(self):
        y = greedydescent.fit_estimator((0, 0), (1, 1), (2, 4), 3)
        self.assertEqual(y, 9)

    def test_not_so_simple(self):
        y = greedydescent.fit_estimator((-2, 20), (1, -7), (2, 8), 0.5)  # 6 x^2 - 3 x - 10
        self.assertEqual(y, -10.0)

    def test_deferral_to_linear(self):
        y = greedydescent.fit_estimator((1, None), (1, 1), (2, 2), 3)
        self.assertEqual(y, 3)


class ComputeNeighboringXs(unittest.TestCase):

    def test_simple(self):
        next_x = greedydescent.compute_next_x([0, 0], 1, 1)
        self.assertEqual(next_x, [0, 1])

    def test_negative(self):
        next_x = greedydescent.compute_next_x([0, 0], 1, -1)
        self.assertEqual(next_x, [0, -1])


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



import unittest
import greedydescent


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



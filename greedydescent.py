from GreedyDescentNode import GreedyDescentNode
from typing import Callable, Union, Tuple, Iterable, Optional
import ComputerAndEstimator

# define 'scalar' to be 'float or int'. For now restricted to 'int'. Scaling can later be implemented to include floats
Scalar = Union[float, int]
# define vector to mean N dimensional scalar vector
Vector = Tuple[Scalar]
# define a type to signify the type of the cost
TCost = float
# define a type to signify the type of the error
TError = float
Coordinate = Tuple[Vector, Optional[TError]]


def minimize(error_computer: ComputerAndEstimator,
             seeds,
             cost_heuristic: Callable[[Vector], TCost],
             abort: Callable[[int, TError, int], bool]=None,
             debug: Callable[[Vector], object]=None,
             weigh=None):
    """
    :param error_computer: An ComputerAndEstimator which caches and computes the error to minimize.
    :param seeds: The iterable of initial vectors in the parameter space
    :param cost_heuristic: A function that takes x and estimates the cost of computing the error at x
    :param abort:
    :param debug:
    :param weigh:
    """

    weigh = weigh if weigh is not None else lambda error_, cost_, x_: error_.weigh(cost_, x_)

    open_list = []      # list of GreedyDescentNodes (containing a vector and scalar, the result of weigh)
    closed_list = set()
    minimum_bias = None
    iterations = 0
    consecutive_higher = 0  # the number of consecutive times compute_error(current) was higher than its minimum so far

    for seed in seeds:
        if seed not in open_list:
            open_list.append(GreedyDescentNode(seed, 0))
    open_list.sort()

    while len(open_list) != 0:
        current = open_list.pop(0).x
        if debug is not None:
            debug(current)
        error = error_computer.compute(current)
        closed_list.add(current)

        xdd_list = get_neighbors(current)  # xdd stands for vector, dimension, direction
        for x, dimension, direction in xdd_list:
            if x not in closed_list:
                estimated_error = error_computer.estimate(x, dimension, direction)
                estimated_cost = cost_heuristic(x)
                f = weigh(estimated_error, estimated_cost, x)  # means weighted cost/loss
                if x in open_list:
                    open_list[open_list.index(x)] = GreedyDescentNode(x, min(f, open_list[open_list.index(x)].cost))
                else:
                    open_list.append(GreedyDescentNode(x, f))
        open_list.sort()  # PERF: could be omitted through heap structure

        yield GreedyDescentNode(current, error)

        if minimum_bias is None or error < minimum_bias:
            minimum_bias = error
            consecutive_higher = 0
        else:
            consecutive_higher = consecutive_higher + 1

        if abort is not None and abort(iterations, minimum_bias, consecutive_higher):
            break
        iterations = iterations + 1


def get_neighbors(x) -> (Vector, int, int):
    for i in range(0, len(x)):
        yield compute_next_x(x, i, 1), i, 1
        yield compute_next_x(x, i, -1), i, -1


def compute_next_x(x, dimension, direction) -> Vector:
    assert direction in [-2, -1, 1, 2]
    assert 0 <= dimension < len(x), "0 <= (dimension = %d) < (len(x) = %d) must hold" % (dimension, len(x))

    result = list(x)
    result[dimension] = compute_next_x_at_d(x, dimension, direction)
    result = tuple(result)  # type: Vector
    return result


def compute_next_x_at_d(x, dimension: int, direction):
    return x[dimension] + compute_dimensional_dx(x, dimension, direction)


def compute_dimensional_dx(x, dimension, direction):
    """
Computes the magnitude of dx, which is in the specified dimension and direction
    """
    assert direction in [-2, -1, 1, 2]
    assert 0 <= dimension < len(x)

    return direction  # for now only integer increment TODO: implement scaling to float

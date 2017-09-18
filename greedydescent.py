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
        for x, dx in xdd_list:
            if x not in closed_list:
                estimated_error = error_computer.estimate(x, dx)
                if estimated_error is not None:
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


def get_neighbors(v) -> (Vector, int, int):
    for dimension in v.dimensions:
        for step in [1, -1]:
            new_v = v.step(dimension, step)
            if new_v is not None:
                yield new_v, (dimension, step)





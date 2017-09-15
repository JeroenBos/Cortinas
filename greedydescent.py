from GreedyDescentNode import GreedyDescentNode
from typing import Callable, Union, Tuple, Iterable, Optional

# define 'scalar' to be 'float or int'. For now restricted to 'int'. Scaling can later be implemented to include floats
Scalar = Union[float, int]
# define vector to mean N dimensional scalar vector
Vector = Tuple[Scalar]
# define a type to signify the type of the cost
TCost = float
# define a type to signify the type of the error
TError = float
Coordinate = Tuple[Vector, Optional[TError]]


def minimize(compute_error: Callable[[Vector, bool], Optional[TError]],
             seeds: Iterable[Vector],
             cost_heuristic: Callable[[Vector], TCost],
             weigh: Callable[[Optional[TError], TCost, Vector], float],
             estimate_error: Callable[[Coordinate, Coordinate, Coordinate, Vector, int], Optional[TError]],
             abort: Callable[[int, TError, int], bool]=None,
             debug: Callable[[Vector], object]=None):
    """
    :param compute_error: A function taking x and must_compute, indicating whether the result should be calculated
                    or whether a cached result suffices.
                  This function computes the error to minimize
    :param seeds: The iterable of initial vectors in the parameter space
    :param cost_heuristic: A function that takes x and estimates the cost of computing the error at x
    :param weigh: A function that takes 3 parameters: estimated_error, estimated_cost, x
                    that weighs the costs of computing the error at x against the estimated error at x
    :param estimate_error:
    :param abort:
    :param debug:
    """

    def cached_error(x_): return compute_error(x_, False)
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
        error = compute_error(current, True)
        closed_list.add(current)

        xdd_list = get_neighbors(current)  # xdd stands for vector, dimension, direction
        for x, dimension, direction in xdd_list:
            if x not in closed_list:
                estimated_loss = fit_loss(x, dimension, direction, cached_error, estimate_error)
                estimated_cost = cost_heuristic(x)
                f = weigh(estimated_loss, estimated_cost, x)  # means weighted cost/loss
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


def fit_loss(x, dimension, direction, cached_error: Callable[[Vector], Optional[TError]], error_estimator):
    """
Estimates L at x
    :param x:
    :param dimension: the direction of dx that resulted in x
    :param direction:
    :param cached_error:
    :param error_estimator:
    :return:
    """
    assert direction in [-1, 1]
    assert 0 <= dimension < len(x)

    def cached_loss(direction_): return (compute_next_x(x, dimension, direction_),
                                         cached_error(compute_next_x(x, dimension, direction_)))
    if direction == 1:
        c1 = cached_loss(-2 * direction)
        c2 = cached_loss(-direction)
        c3 = cached_loss(direction)
        return fit_estimator(c1, c2, c3, x, dimension, error_estimator)
    else:
        return fit_estimator(cached_loss(direction),
                             cached_loss(-direction),
                             cached_loss(-2 * direction),
                             x,
                             dimension,
                             error_estimator)


def compute_next_x(x: Vector, dimension, direction) -> Vector:
    assert direction in [-2, -1, 1, 2]
    assert 0 <= dimension < len(x), "0 <= (dimension = %d) < (len(x) = %d) must hold" % (dimension, len(x))

    result = list(x)
    result[dimension] = compute_next_x_at_d(x, dimension, direction)
    result = tuple(result)  # type: Vector
    return result


def compute_next_x_at_d(x: Vector, dimension: int, direction):
    return x[dimension] + compute_dimensional_dx(x, dimension, direction)


def compute_dimensional_dx(x, dimension, direction):
    """
Computes the magnitude of dx, which is in the specified dimension and direction
    """
    assert direction in [-2, -1, 1, 2]
    assert 0 <= dimension < len(x)

    return direction  # for now only integer increment TODO: implement scaling to float


def fit_estimator(coordinate1, coordinate2, coordinate3, v, dimension, estimator):
    v1, error1 = coordinate1
    v2, error2 = coordinate2
    v3, error3 = coordinate3

    if error1 is None:
        return fit_estimator2(coordinate2, coordinate3, v, dimension, estimator)
    if error2 is None:
        return fit_estimator2(coordinate1, coordinate3, v, dimension, estimator)
    if error3 is None:
        return fit_estimator2(coordinate1, coordinate2, v, dimension, estimator)

    return estimator.estimate3(coordinate1, coordinate2, coordinate3, v, dimension)


def fit_estimator2(coordinate1, coordinate2, v, dimension, estimator):
    v1, error1 = coordinate1
    v2, error2 = coordinate2

    if error1 is None:
        return None
    if error2 is None:
        return None

    return estimator.estimate2(coordinate1, coordinate2, v, dimension)

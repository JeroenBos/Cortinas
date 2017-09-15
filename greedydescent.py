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


def minimize(compute_error: Callable[[Vector, bool], Optional[TError]],
             seeds: Iterable[Vector],
             cost_heuristic: Callable[[Vector], TCost],
             weigh: Callable[[Optional[TError], TCost, Vector], float],
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
                estimated_loss = fit_loss(x, dimension, direction, cached_error)
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


def fit_loss(x, dimension, direction, cached_error: Callable[[Vector], Optional[TError]]):
    """
Estimates L at x
    :param x:
    :param dimension: the direction of dx that resulted in x
    :param direction:
    :param cached_error:
    :return:
    """
    assert direction in [-1, 1]
    assert 0 <= dimension < len(x)

    def cached_loss(direction_): return (compute_next_x_at_d(x, dimension, direction_),
                                         cached_error(compute_next_x(x, dimension, direction_)))
    if direction == 1:
        c1 = cached_loss(-2 * direction)
        c2 = cached_loss(-direction)
        c3 = cached_loss(direction)
        return fit_estimator(c1, c2, c3, x[dimension])
    else:
        return fit_estimator(cached_loss(direction), cached_loss(-direction), cached_loss(-2 * direction), x[dimension])


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


def are_unique(elements):
    seen = set()
    return not any(i in seen or seen.add(i) for i in elements)


def fit_estimator(coordinate1, coordinate2, coordinate3, x):
    """
Fits a parabola to the specified coordinates and estimates the y value at the specified x value
    :param coordinate1:
    :param coordinate2:
    :param coordinate3:
    :param x:
    :return:
    """
    x1, y1 = coordinate1
    x2, y2 = coordinate2
    x3, y3 = coordinate3
    if y1 is None:
        return linear_fit_estimator(x2, y2, x3, y3, x)
    if y2 is None:
        return linear_fit_estimator(x1, y1, x3, y3, x)
    if y3 is None:
        return linear_fit_estimator(x1, y1, x2, y2, x)
    assert are_unique([x1, x2, x3, x]), ValueError('The specified x values must differ')

    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d
    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / d
    return a * x * x + b * x + c


def linear_fit_estimator(x1, y1, x2, y2, x):
    if y1 is None:
        return None
    if y2 is None:
        return None
    if x1 == x2:
        raise ValueError('The specified x values must differ')

    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a * x + b

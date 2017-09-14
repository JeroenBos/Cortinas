import copy


# define 'scalar' to be 'float or int'. For now restricted to 'int'. Scaling can later be implemented to include floats
# define vector to mean N dimensional scalar vector
# I want to define a function that takes
#
# - a function L to minimize which takes a vector and a boolean indicating whether only cached values should be returned
#                            and which returns a scalar option (only None if must_compute = False)
# - a seed mechanism to choose the initial (one or many) vectors
# - a function C which takes x (a vector) that estimates the cost of evaluating L at x
# - a function F which takes x, a gradient option vector dL in L-space and cost c and returns a comparable,
#     where c is evaluated at x and d[i] = L(x - e_i) - L(x - 2e_i)
#     where e_i is the unit step in the ith dimension. This means L(x - e_i) must always already have been computed
#                                                      but L(x - 2e_i) not necessarily, in which case d[i] = None
#
# the output of this function is the enumerable of pairs of vectors and non-option results of L
#
# Implementation
# I should probably implement this through an A* algorithm, but where there is not cost in moving
# but where there is a cost in exploring.
# More concretely, the comparable returned by F is what in A* would be called F
# Hmm, then there's no use for G. So it becomes a Dijkstra's algorithm
# so pseudo-algorithm:
# def minimize(L, get_seeds, C, F, abort):
#     open_list = []      # list of GreedyDescentNodes (containing a vector and scalar, the result of C)
#     closed_list = set()
#     minimum_loss = None
#     iterations = 0
#
#     for seed in get_seeds():
#         if seed not in open_list:
#             open_list.append((seed, 0))
#
#     while len(open_list) != 0:
#         current = open_list.pop()
#         closed_list.add(current)
#         x_array = get_neighbors(current)
#         for x in x_array:
#             if x not in open_list[0, :] and x not in closed_list:
#                 loss_x = estimate_loss(L, x)
#                 c = C(x)
#                 f = F(x, loss_x, c)
#                 open_list.append((x, f))
#             open_list.sort(key=lambda open_pair: open_pair[1]) # PERF: could be omitted through heap structure
#         loss = L(current, True)
#         yield loss
#         minimum_loss = min(minimum_loss, loss) if minimum_loss is not None else loss
#         if abort(iterations, minimum_loss):
#             break
#         iterations = iterations + 1
#
#
# def get_neighbors(x):
#     for i in range(0, len(x)):
#         yield compute_next_x(x, i, 1)
#         yield compute_next_x(x, i, -1)


def estimate_loss(x, dimension, direction, loss):
    """
Estimates L at x
    :param x:
    :param dimension: the direction of dx that resulted in x
    :param direction:
    :param loss:
    :return:
    """
    assert direction in [-1, 1]
    assert 0 <= dimension < len(x)

    def cached_loss(direction_): (compute_next_x(x, dimension, direction_),
                                  loss(compute_next_x(x, dimension, direction_), False))
    if direction == 1:
        return fit_estimator(cached_loss(-2 * direction), cached_loss(-direction), cached_loss(direction), x)
    else:
        return fit_estimator(cached_loss(direction), cached_loss(-direction), cached_loss(-2 * direction), x)


def compute_next_x(x, dimension, direction):
    assert direction in [-1, 1]
    assert 0 <= dimension < len(x), "0 <= (dimension = %d) < (len(x) = %d) must hold" % (dimension, len(x))

    result = copy.copy(x)
    dx_dimension = compute_dimensional_dx(x, dimension, direction)
    result[dimension] = result[dimension] + dx_dimension
    return result


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

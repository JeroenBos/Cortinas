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
# minimize(L, get_seeds, C, F)
#    open = []      # list of tuples of vector and scalar (result of C)
#    closed = []
#
#    for seed in get_seeds():
#        if not open.contains(seed):
#            open.add((seed, 0))
#
#    minimal_L = infinity
#    iterations = 0
#    while len(open != 0)
#        current = open.pop()
#        closed.add(current)
#        x_array = get_neighbors(current) # array of tuples of vectors and dL options
#        for x in x_array:
#            if not open.contains(x) && not closed.contains(x)
#                dL = compute_dL(L, x)
#                c = C(x)
#                f = F(x, dL, c)
#                open.add((x, f))
#        open.sort(key=lambda open_pair: open_pair[1]) # could be omitted through heap structure
#        l = L(x, True)
#        yield return l
#
#        minimal_L = min(minimal_L, l)
#        if abort(iterations, minimal_)
#            break
#        iterations = iterations + 1
#
#
#
#
#
# def get_neighbors(x)
#    for i in range(0, len(x)):
#        yield compute_next_x(x, i, 1)
#        yield compute_next_x(x, i, -1)
#
#
#
# something wrong here. why return None sometimes? d always exist right? it depends on whether it's in open or closed
#
# def compute_next_x(x, dimension, direction):
#    if direction != 1 && direction != -1
#        raise ArgumentError
#    if dimension >= len(x)
#        raise ArgumentError
#
#    result = copy(x)
#    dx_dimension = compute_dimensional_dx(x, dimension, direction)
#    result[dimension] = result[dimension] + dx_dimension
#    return result
#
# computes the magnitude of dx, which is in the specified dimension and direction
# def compute_dimensional_dx(x, dimension, direction):
#    if direction != 1 && direction != -1 && direction != 2 && direction != -2
#        raise ArgumentError
#    if i >= len(x)
#        raise ArgumentError
#
#    return direction # for now only integer increment TODO: implement scaling to float
#
#
# # estimates L at x
# # param direction: the direction of dx that resulted in x
# def estimate_L(x, dimension, direction):
#    if direction != 1 && direction != -1
#        raise ArgumentError
#    if dimension >= len(x)
#        raise ArgumentError
#
#    cached_L = direction => x_ = compute_next_x(x, dimension, direction)
#                            return x, L(x, False)
#    if direction == 1:
#        return fit_estimator(cached_L(-2 * direction), cached_L(-direction), cached_L(direction))
#    else
#        return fit_estimator(cached_L(direction), cached_L(-direction), cached_L(-2 * direction))
#
#
# # fits a parabola to the specified points and
# def fit_estimator((x1, y1), (x2, y2), (x3, y3), x):
#
#    if y1 == None:
#        return linear_fit_estimator((x2, y2), (x3, y3), x)
#    if y2 == None:
#        return linear_fit_estimator((x1, y1), (x3, y3), x)
#    if y3 == None:
#        return linear_fit_estimator((x1, y1), (x2, y2), x)
#
#    D = (x1 - x2) * (x1 - x3) * (x2 - x3);
#    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / D;
#    b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / D;
#    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / D;
#
#    return a * x * x + b * x + c
#
#
# def linear_fit_estimator((x1, y1), (x2, y2), x)
#
#    if y1 == None:
#        return None
#    if y2 == None:
#        return None
#
#    a = (y2 - y1) / (x2 - x1)
#    b = y1 - a * x1
#
#    return a * x + b

# The float in the scalar must probably become a decimal because I'm gonna need equality comparison on it



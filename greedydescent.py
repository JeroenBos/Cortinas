# define 'scalar' to be 'float or int'.
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
#        x_dL_array = get_neighbors(current) # array of tuples of vectors and dL options
#        for x, dL in x_dL_array:
#            if not open.contains(x) && not closed.contains(x)
#                c = C(x)
#                f = F(x, dL, c)
#                open.add((x, f))
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
#        yield compute_d(x, i, 1)
#        yield compute_d(x, i, -1)
#
#
#
# something wrong here. why return None sometimes? d always exist right? it depends on whether it's in open or closed
#
# def compute_d(x, i, direction):
#    if direction != 1 && direction != -1
#        raise ArgumentError
#    if i >= len(x)
#        raise ArgumentError
#
#    if L(diff(x, i, -1), False) != None:
#        if L(diff(x, i, -2), False) == None:
#            return None
#        else
#            d = copy(x) # array of scalar options
#            d[i] = L(diff(x, i, -2), False) - L(diff(x, i, -1), False)
#            return d
#    elif L(diff(x, i, 1), False) != None:
#        if L(diff(x, i, 2), False) == None:
#            return None
#        elif
#            d = copy(x) # array of scalar options
#            d[i] = L(diff(x, i, 2), False) - L(diff(x, i, 1), False)
#            return d
#    else
#        return None
#
# def diff(x, dimension, direction):
#    if direction != 1 && direction != -1 && direction != 2 && direction != -2
#        raise ArgumentError
#    if i >= len(x)
#        raise ArgumentError
#    raise NotImplementedError

# The float in the scalar must probably become a decimal because I'm gonna need equality comparison on it



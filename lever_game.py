import numpy as np
from scipy.special import softmax
from shgo import SHGO
import timeit


import dgraph


def fun(theta):
    """
    Computes the return of the one-step lever game.
    See: https://proceedings.mlr.press/v119/hu20a.html
    :param theta: Parameters of the self-play policy.
    :return: The negative return, used for minimization.
    """
    policy = theta

    levers = np.eye(NDIM)
    levers[0, 0] = 0.6
    levers[1, 1] = 0.8
    levers[2, 2] = 0.8
    t1, t2 = softmax(policy * TAU), softmax(policy * TAU)
    r = np.outer(t1, t2) * levers
    return -r.sum()


if __name__ == "__main__":
    NDIM = 10
    TAU = 10

    bounds = [(0.0, 1.0), ] * NDIM  # boundaries of theta

    start_time = timeit.default_timer()
    shc = SHGO(
        fun,
        bounds,
        n=int(1e3),
        workers=1,
        sampling_method='simplicial'
    )
    shc.iterate_all()

    print(f'Took {timeit.default_timer()-start_time:.2f} seconds.')

    for i, (value, solution) in enumerate(zip(shc.res.funl, shc.res.xl)):
        print(f'Result: {i}, \tReturn: {value:.2f}, \tSolution: {solution}')

    # plot graph
    HC = shc.HC
    minima = [HC.V[v] for v in HC.V.cache if HC.V[v].minimiser()]
    g = dgraph.construct_network_graph(minima, fun)
    dtree = dgraph.make_disconnectivity_tree(g)
    dtree.plot()




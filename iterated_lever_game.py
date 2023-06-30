from shgo import shgo
from tgo import tgo
import numpy as np
import timeit
from scipy.special import softmax
from itertools import combinations, permutations, product


def roll_axis(A, r):
    res = A.copy()
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]

    return res[rows, column_indices]


def permute_state_batch(state, perm):
    s1, s2 = state[:, :NDIM], state[:, NDIM:]

    obs = np.concatenate([roll_axis(s1, perm), roll_axis(s2, perm)], axis=1)
    assert obs.shape == state.shape
    return obs


def permute_act_batch(act, perm):
    a1, a2 = act[:, :NDIM], act[:, NDIM:]

    act_ = np.concatenate([roll_axis(a1, -perm), roll_axis(a2, -perm)], axis=1)
    assert act.shape == act_.shape
    return act_


def soft_batch(x, tau=1.):
    '''
      return: softmax
      param: x of shape [batch, ndim]
    '''
    e_x = np.exp(x / tau)
    e_x = np.divide(e_x.reshape(NEP, NDIM), e_x.sum(axis=1).reshape(NEP, 1))
    return e_x


def round_batch(theta, state1, state2, i, j, return_actions=False):
    levers = np.repeat(np.eye(NDIM).reshape(1, NDIM, NDIM), NEP, 0)
    I = np.repeat(np.eye(NDIM).reshape(1, NDIM, NDIM), NEP, 0)

    # permute
    if OTHER_PLAY:
        obs2 = permute_state_batch(state2, i)
        obs1 = permute_state_batch(state1, j)

    theta1 = theta[:, :NDIM]
    theta2 = theta[:, :NDIM]

    l1 = np.matmul(theta1, obs1.reshape(NEP, NDIM * 2, 1))
    l2 = np.matmul(theta2, obs2.reshape(NEP, NDIM * 2, 1))
    assert l1.shape == l2.shape == (NEP, NDIM, 1)

    p1, p2 = soft_batch(l1 * TAU), soft_batch(l2 * TAU)
    assert p1.shape == p2.shape == (NEP, NDIM)

    # random choice with prob p1 and p2
    a1 = (p1.cumsum(1) > np.random.rand(NEP)[:, None]).argmax(1)
    a1hot = I[np.arange(NEP), a1]

    a2 = (p2.cumsum(1) > np.random.rand(NEP)[:, None]).argmax(1)
    a2hot = I[np.arange(NEP), a2]
    assert a1hot.sum() == a2hot.sum() == NEP

    # re-permute
    if OTHER_PLAY:
        a2hot = permute_act_batch(a2hot, i)
        a1hot = permute_act_batch(a1hot, j)

    # joint policy
    outer = np.matmul(a1hot.reshape(NEP, NDIM, 1), a2hot.reshape(NEP, 1, NDIM))
    # reward
    r = np.einsum('ikl, ikl -> ikl', outer, levers)
    # sum over actions (joint matrix)
    r = r.sum((1, 2))

    if return_actions:
        return r, a1hot, a2hot, a1, a2
    else:
        return r, a1hot, a2hot


def fun(pi):
    """
      batch implementation of game with NEP number of episodes
      pi: array of shape 2*NDIM x 2*NDIM
        first NDIM rows are for player 1, second NDIM rows are for player 2
        state of player 1 is (a1, a2), state of player 2 is (a2, a1)
      return: return
    """

    a1hot, a2hot = np.zeros(NDIM), np.zeros(NDIM)

    state = np.repeat(np.concatenate([a1hot, a2hot]).reshape(1, -1), NEP, 0)
    assert state.shape == (NEP, NDIM * 2)
    state1 = state2 = state

    theta = np.repeat(pi.reshape(1, NDIM, NDIM * 2), NEP, 0)

    R = np.zeros(NEP)

    i = np.random.choice(NDIM, size=NEP)
    j = np.random.choice(NDIM, size=NEP)
    for t in range(TMAX):
        # logits
        r, a1hot, a2hot = round_batch(theta, state1, state2, i, j)

        assert len(r) == NEP
        R += r

        # a1hot shape: [NEP, NDIM]
        # state1 shape: [NEP, 2*NDIM]
        assert a1hot.shape == (NEP, NDIM)
        state1 = np.concatenate([a1hot, a2hot], -1)
        state2 = np.concatenate([a2hot, a1hot], -1)

    return -R.sum() / NEP / TMAX


def example_solution():
    pi = np.hstack([np.eye(NDIM), np.eye(NDIM)])
    value = fun(pi)
    print(f'Return: {value:.2f}, Solution: \n{pi}')

def example_shgo():
    # TAU = 20.
    bounds = [(0., 1.)] * NDIM * 2 * NDIM  # boundaries of theta

    start_time = timeit.default_timer()
    res = shgo(
        fun,
        bounds,
        n=N,  # Possible scales should be benchmarked for your application, 1e8 to 1e9 would be ideal.
        workers=WORKERS
    )
    print(f'Took {timeit.default_timer() - start_time:.2f} seconds.')
    TOP_K = 10
    for i, (value, solution) in enumerate(zip(res.funl, res.xl)):
        print(f'Result: {i}, \tReturn: {value:.2f}, Solution: \n{np.around(solution, 2).reshape(NDIM, -1)}')
        if i == TOP_K: break


def example_tgo():
    # TAU = 20.
    bounds = [(0., 1.)] * NDIM * 2 * NDIM  # boundaries of theta

    start_time = timeit.default_timer()
    res = tgo(
        fun,
        bounds,
        n=N,  # Possible scales should be benchmarked for your application, 1e8 to 1e9 would be ideal.
    )
    print(f'Took {timeit.default_timer() - start_time:.2f} seconds.')
    TOP_K = 10
    for i, (value, solution) in enumerate(zip(res.funl, res.xl)):
        print(f'Result: {i}, \tReturn: {value:.2f}, Solution: \n{np.around(solution, 2).reshape(NDIM, -1)}')
        if i == TOP_K: break


if __name__ == "__main__":
    TMAX = 5
    NDIM = 3
    TAU = 5.
    OTHER_PLAY = True
    NEP = int(1e3)

    # These can be adjusted
    N = int(1e3)
    WORKERS = 1

    example_solution()
    example_shgo()
    example_tgo()



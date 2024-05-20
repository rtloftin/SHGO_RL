"""
This script implements the lever game from https://proceedings.mlr.press/v119/hu20a.html.

Instead of one time step, the game consists of multiple rounds.
"""

from shgo import shgo
from tgo import tgo
import numpy as np
import timeit


def roll_axis(A, r):
    """
    Roll the elements of each row in a 2D array by a given amount.

    This function takes a 2D array `A` and a 1D array `r` where each element
    in `r` represents the number of positions by which to roll the corresponding
    row in `A`.

    Parameters:
    A (np.ndarray): A 2D numpy array where each row will be rolled.
    r (np.ndarray): A 1D numpy array of integers representing the number of positions
                    to roll each row in `A`.

    Returns:
    np.ndarray: A new 2D array with the rows of `A` rolled according to `r`.

    Example:
    >>> A = np.array([[0, 0, 1], [0, 1, 0]])
    >>> r = np.array([1, 2])
    >>> roll_axis(A, r)
    array([[1, 0, 0],
           [1, 0, 0]])
    """
    res = A.copy()
    rows, column_indices = np.ogrid[: A.shape[0], : A.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]

    return res[rows, column_indices]


def permute_state_batch(state, perm):
    """
    Permute the rows of a batched state array according to a given permutation.

    This function takes a 2D state array `state` and a 1D array `perm` representing
    the permutation for each row. The state array is assumed to have a dimensionality
    split into two equal parts along the second axis. Each part is permuted separately
    and then concatenated to form the resulting array.

    Parameters:
    state (np.ndarray): A 2D numpy array with shape (batch_size, 2*NDIM), where each row
                        represents a batched state consisting of two parts.
    perm (np.ndarray): A 1D numpy array of integers representing the number of positions
                       by which to permute each corresponding row in the state array.

    Returns:
    np.ndarray: A new 2D array with the same shape as `state`, with the rows permuted
                according to `perm`.

    Example:
    >>> state = np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0]])
    >>> perm = np.array([1, 1])
    >>> permute_state_batch(state, perm)
    array([[0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1]])
    """
    s1, s2 = state[:, :NDIM], state[:, NDIM:]

    obs = np.concatenate([roll_axis(s1, perm), roll_axis(s2, perm)], axis=1)
    assert obs.shape == state.shape
    return obs


def permute_act_batch(act, perm):
    """
    Permute the rows of a batched action array according to a given permutation.

    This function takes a 2D action array `act` and a 1D array `perm` representing
    the permutation for each row. The action array is assumed to have a dimensionality
    split into two equal parts along the second axis. Each part is permuted separately
    using the negative of the provided permutation and then concatenated to form the
    resulting array.

    Parameters:
    act (np.ndarray): A 2D numpy array with shape (batch_size, 2*NDIM), where each row
                      represents a batched action consisting of two parts.
    perm (np.ndarray): A 1D numpy array of integers representing the number of positions
                       by which to permute each corresponding row in the action array.

    Returns:
    np.ndarray: A new 2D array with the same shape as `act`, with the rows permuted
                according to the negative of `perm`.

    Example:
    >>> act = np.array([[0, 0, 1], [0, 0, 1]])
    >>> perm = np.array([2, 0])
    >>> permute_act_batch(act, perm)
    array([[1, 0, 0],
           [0, 0, 1]])
    """
    a1, a2 = act[:, :NDIM], act[:, NDIM:]

    act_ = np.concatenate([roll_axis(a1, -perm), roll_axis(a2, -perm)], axis=1)
    assert act.shape == act_.shape
    return act_


def soft_batch(x, tau=1.0):
    """
    Compute the softmax of each row in a batched input array.

    This function takes a 2D array `x` and a temperature parameter `tau`, and computes the
    softmax for each row. The softmax function is computed as exp(x / tau) normalized by
    the sum of exponentials over each row. The temperature parameter `tau` controls the
    sharpness of the distribution; lower values of `tau` make the distribution more peaky.

    Parameters:
    x (np.ndarray): A 3D numpy array of shape [batch, ndim, 1], where each row represents a
                    set of logits.
    tau (float): A temperature parameter that controls the sharpness of the softmax distribution.
                 Default is 1.0.

    Returns:
    np.ndarray: A 2D numpy array of shape [batch, ndim], containing the softmax probabilities
                for each row.

    Example:
    >>> x = np.array([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    >>> soft_batch(x, tau=1.0)
    array([[0.09003057, 0.24472847, 0.66524096],
           [0.09003057, 0.24472847, 0.66524096]])
    """
    e_x = np.exp(x / tau)
    e_x = np.divide(e_x.reshape(NEP, NDIM), e_x.sum(axis=1).reshape(NEP, 1))
    return e_x


def round_batch(theta, state1, state2, i, j):
    """
    Execute a single round of the batched iterated lever game.

    This function performs one round of the game for a batch of agents. It computes the
    actions and rewards for each agent based on their state and strategy parameters,
    considering possible permutations of states and actions.

    Parameters:
    theta (np.ndarray): A 3D numpy array of shape (NEP, NDIM, 2*NDIM) representing the strategy
                        parameters for each agent.
    state1 (np.ndarray): A 2D numpy array of shape (NEP, 2*NDIM) representing the state of
                         the first set of agents.
    state2 (np.ndarray): A 2D numpy array of shape (NEP, 2*NDIM) representing the state of
                         the second set of agents.
    i (np.ndarray): A 1D numpy array representing the permutation indices for the first set
                    of agents.
    j (np.ndarray): A 1D numpy array representing the permutation indices for the second set
                    of agents.

    Returns:
    tuple: A tuple containing:
        - r (np.ndarray): A 1D numpy array of shape (NEP,) representing the rewards for each agent.
        - a1hot (np.ndarray): A 2D numpy array of shape (NEP, NDIM) representing the one-hot encoded
                              actions for the first agent.
        - a2hot (np.ndarray): A 2D numpy array of shape (NEP, NDIM) representing the one-hot encoded
                              actions for the second agent.

    Example: previous values a1hot, a2hot = [[0. 0. 1.], [0. 1. 0.]], [[1. 0. 0.], [0. 1. 0.]]
    >>> theta = np.random.rand(NEP, NDIM, 2 * NDIM)
    >>> state1 = [[0. 0. 1. 1. 0. 0.], [0. 1. 0. 0. 1. 0.]]
    >>> state2 = [[1. 0. 0. 0. 0. 1.], [0. 1. 0. 0. 1. 0.]]
    >>> i = [0 0]
    >>> j = [2 0]
    >>> r, a1hot, a2hot = round_batch(theta, state1, state2, i, j)

    """
    levers = np.repeat(np.eye(NDIM).reshape(1, NDIM, NDIM), NEP, 0)
    eye = np.repeat(np.eye(NDIM).reshape(1, NDIM, NDIM), NEP, 0)

    # permute
    if OTHER_PLAY:
        obs2 = permute_state_batch(state2, i)
        obs1 = permute_state_batch(state1, j)
    else:
        obs2 = state2
        obs1 = state1

    theta1 = theta[:, :NDIM]
    theta2 = theta[:, :NDIM]

    l1 = np.matmul(theta1, obs1.reshape(NEP, NDIM * 2, 1))
    l2 = np.matmul(theta2, obs2.reshape(NEP, NDIM * 2, 1))
    assert l1.shape == l2.shape == (NEP, NDIM, 1)

    p1, p2 = soft_batch(l1 * TAU), soft_batch(l2 * TAU)
    assert p1.shape == p2.shape == (NEP, NDIM)

    # random choice with prob p1 and p2
    a1 = (p1.cumsum(1) > np.random.rand(NEP)[:, None]).argmax(1)
    a1hot = eye[np.arange(NEP), a1]

    a2 = (p2.cumsum(1) > np.random.rand(NEP)[:, None]).argmax(1)
    a2hot = eye[np.arange(NEP), a2]
    assert a1hot.sum() == a2hot.sum() == NEP

    # re-permute
    if OTHER_PLAY:
        a2hot = permute_act_batch(a2hot, i)
        a1hot = permute_act_batch(a1hot, j)

    # joint policy
    outer = np.matmul(a1hot.reshape(NEP, NDIM, 1), a2hot.reshape(NEP, 1, NDIM))
    # reward
    r = np.einsum("ikl, ikl -> ikl", outer, levers)
    # sum over actions (joint matrix)
    r = r.sum((1, 2))

    return r, a1hot, a2hot


def fun(pi):
    """
    Batch implementation of the lever game with NEP number of episodes.

    This function simulates the iterated lever game for a batch of episodes. It updates
    the states and actions of two players over multiple rounds and computes the average
    reward.

    Parameters:
    pi (np.ndarray): A 1D numpy array of shape (NDIM*NDIM) representing the policy matrix.
                     If SELF_PLAY is False, shape is (NDIM*NDIM*2).
                     The elements are for player 1, and for player 2.
                     The state of player 1 is (a1, a2), and the state of player 2 is (a2, a1).

    Returns:
    float: The negative average reward over all episodes and rounds.

    Example:
    >>> pi = np.random.rand(2 * NDIM, 2 * NDIM)
    >>> result = fun(pi)
    """
    a1hot, a2hot = np.zeros(NDIM), np.zeros(NDIM)

    state = np.repeat(np.concatenate([a1hot, a2hot]).reshape(1, -1), NEP, 0)
    assert state.shape == (NEP, NDIM * 2)
    state1 = state2 = state

    if SELF_PLAY:
        pi = pi.reshape(NDIM, NDIM)
        pi = np.hstack([pi, pi])
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

    f = -R.sum() / NEP / TMAX
    return f


def example_solution():
    """
    Demonstrate an example solution for the iterated lever game.

    This function constructs a policy matrix `pi` based on the value of `SELF_PLAY`.
    If `SELF_PLAY` is True, the policy matrix is created as a NDIM*NDIM matrix, used for selecting act1 and act2.
    Otherwise, it concatenates two matrices side-by-side for the case act1 is selected by first NDIM*NDIM elements,
    and act2 by second NDIM*NDIM elements. The function then calculates the game
    value using the `fun` function and prints the result.
    """
    if SELF_PLAY:
        pi = np.eye(NDIM).reshape(-1)
    else:
        pi = np.hstack([np.eye(NDIM), np.eye(NDIM)]).reshape(-1)
    value = fun(pi)
    print(f"Return: {value:.2f}, Solution: \n{pi}")


def example_shgo():
    """Demonstrate a solution for the iterated lever game found by SHGO global optimization algorithm."""
    # TAU = 20.
    if SELF_PLAY:
        bounds = [(0.0, 1.0)] * NDIM * NDIM
    else:
        bounds = [(0.0, 1.0)] * NDIM * 2 * NDIM  # boundaries of theta

    start_time = timeit.default_timer()
    res = shgo(
        fun,
        bounds,
        n=N,
        workers=WORKERS,
    )
    print(f"Took {timeit.default_timer() - start_time:.2f} seconds.")
    TOP_K = 10
    for i, (value, solution) in enumerate(zip(res.funl, res.xl)):
        print(
            f"Result: {i}, \tReturn: {value:.2f}, Solution: "
            f"\n{np.around(solution, 2).reshape(NDIM, -1)}"
        )
        if i == TOP_K:
            break


def example_tgo():
    """Demonstrate a solution for the iterated lever game found by SHGO global optimization algorithm."""
    # TAU = 20.
    if SELF_PLAY:
        bounds = [(0.0, 1.0)] * NDIM * NDIM
    else:
        bounds = [(0.0, 1.0)] * NDIM * 2 * NDIM  # boundaries of theta

    start_time = timeit.default_timer()
    res = tgo(
        fun,
        bounds,
        n=N,
    )
    print(f"Took {timeit.default_timer() - start_time:.2f} seconds.")
    TOP_K = 10
    for i, (value, solution) in enumerate(zip(res.funl, res.xl)):
        print(
            f"Result: {i}, \tReturn: {value:.2f}, Solution: "
            f"\n{np.around(solution, 2).reshape(NDIM, -1)}"
        )
        if i == TOP_K:
            break


if __name__ == "__main__":
    TMAX = 5
    NDIM = 3
    TAU = 5.0
    OTHER_PLAY = True
    # policy acts same to actions of opponent as to its own actions
    SELF_PLAY = True
    NEP = int(1e3)

    # These can be adjusted
    N = int(2e3)
    WORKERS = 1

    example_solution()
    example_shgo()
    example_tgo()

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from shgo import shgo
from tgo import tgo
import timeit


class Actor(object):
    def __init__(self, s_size=3, a_size=1, h_size=4):
        self.na = a_size
        self.ns = s_size
        self.nh = h_size
        # self.set_theta(theta)

    def set_theta(self, theta):
        shapes = [(self.nh, self.ns), (self.na, self.nh)]
        shapes = [shapes[0]] + [(self.nh, self.nh)] * num_hidden + [shapes[-1]]
        start = 0
        self.thetas = []
        for i, (dx, dy) in enumerate(shapes):
            size = shapes[i][0] * shapes[i][1]
            end = start + size
            self.thetas.append(theta[start:end].reshape(dx, dy))
            start = start + size

    def forward(self, x):
        for theta in self.thetas:
            x = np.tanh(theta @ x)
        return x

    def act(self, state):
        return 2 * self.forward(state)


def fun(theta, gamma=1.0):
    """
    Estimate return for mountain car game.
    :param theta: Parameters of policy.
    :param gamma: Future discount.
    :return: The estimated return for policy theta.
    """
    R = 0
    actor.set_theta(theta)
    for _ in range(N_EPS):
        rewards = []
        state, _ = env.reset(seed=SEED)
        for _ in range(T_MAX):
            action = actor.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R += sum([a * b ** 2 for a, b in zip(discounts, rewards)])
    # reward of +1 for every step taken
    return R / N_EPS


def evaluate(theta):
    env = gym.make(
        'Pendulum-v1', g=G_FORCE,
        render_mode='rgb_array'
    )
    actor.set_theta(theta)

    state, _ = env.reset(seed=SEED)
    img = plt.imshow(env.render())
    rewards = []
    for _ in range(T_MAX):
        action = actor.act(state)
        img.set_data(env.render())
        plt.axis('off')
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
        plt.pause(.01)

    env.close()
    return sum(rewards)


def example():
    theta = np.array([-1., -1., 1., 1., 1., -1., 1., -1., -1., -1., -1., -1., -1.,
                      -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                      -1., -1., -1., -1., -1., -1.])
    value = evaluate(theta)
    print(f'Return: {value:.2f}, Solution: \n{theta}')


def example_shgo():
    bounds = [(-1, 1) for _ in range((S_SIZE + A_SIZE + num_hidden * H_SIZE) * H_SIZE)]

    start_time = timeit.default_timer()
    res = shgo(
        fun,
        bounds,
        n=N,
        workers=WORKERS
    )
    print(f'Took {timeit.default_timer() - start_time:.2f} seconds.')
    TOP_K = 10
    for i, (value, solution) in enumerate(zip(res.funl, res.xl)):
        print(f'Result: {i}, \tReturn: {value:.2f}, Solution: \n{np.around(solution, 2)}')
        if i == TOP_K: break
    env.close()

    evaluate(res.x)


if __name__ == "__main__":
    S_SIZE = 3
    A_SIZE = 1
    H_SIZE = 4
    num_hidden = 1
    T_MAX = 200
    SEED = None

    # These can be adjusted
    N_EPS = 2
    N = int(1e3)
    G_FORCE = 4.81  # 1.81 will always work, default is 9.81
    WORKERS = 1

    env = gym.make('Pendulum-v1', g=G_FORCE)
    actor = Actor(
        s_size=S_SIZE,
        a_size=A_SIZE,
        h_size=H_SIZE
    )

    example()
    example_shgo()

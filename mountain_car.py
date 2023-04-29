import gym
import matplotlib.pyplot as plt
import numpy as np
from shgo import shgo
import timeit
from scipy.special import softmax


class Actor(object):
    """
    Allowing to change the parameters of the actor's policy externally.
    """
    def __init__(self):
        pass

    def set_theta(self, theta):
        self.theta = theta.reshape(A_SIZE, S_SIZE)

    def fc(self, x):
        return self.theta @ x

    def forward(self, x):
        x = self.fc(x)
        return softmax(x)

    def act(self, state):
        probs = self.forward(state)
        return np.argmax(probs)


def fun(theta, gamma=1.0):
    """
    Estimate return for mountain car game.
    :param theta: Parameters of policy.
    :param gamma: Future discount.
    :return: The estimated return for policy theta.
    """
    R = 0
    for _ in range(N_EPS):
        rewards = []
        state = env.reset()
        actor.set_theta(theta)
        for _ in range(T_MAX):
            action = actor.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R += sum([a * b ** 2 for a, b in zip(discounts, rewards)])
    return R / N_EPS


def evaluate(theta):
    env = gym.make(
        'MountainCar-v0',
        render_mode='single_rgb_array',
        new_step_api=True
    )
    env.reset(seed=1)

    actor = Actor()
    actor.set_theta(theta)

    state = env.reset()
    img = plt.imshow(env.render())
    rewards = []
    while True:
        action = actor.act(state)
        img.set_data(env.render())
        plt.axis('off')
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
        plt.pause(.1)

    env.close()


if __name__ == "__main__":
    S_SIZE = 2
    A_SIZE = 3
    T_MAX = 100
    N_EPS = 2

    bounds = [(0, 1) for _ in range(S_SIZE * A_SIZE)]  # boundaries of theta

    env = gym.make('MountainCar-v0', new_step_api=True)
    actor = Actor()

    start_time = timeit.default_timer()
    result = shgo(
        fun,
        bounds,
        n=int(1e3),
        workers=1,
        sampling_method='simplicial',
    )
    print(f'Took {timeit.default_timer()-start_time:.2f} seconds.')
    env.close()

    evaluate(result.x)
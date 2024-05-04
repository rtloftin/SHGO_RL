import gymnasium as gym
import matplotlib.pyplot as plt
from shgo import shgo
import timeit


from actor import Actor


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
        state, _ = env.reset()
        actor.set_theta(theta)
        for _ in range(T_MAX):
            action = actor.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        discounts = [gamma**i for i in range(len(rewards) + 1)]
        R += sum([a * b**2 for a, b in zip(discounts, rewards)])
    # reward of +1 for every step taken
    return -R / N_EPS


def evaluate(theta):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset(seed=1)

    actor = Actor(s_size=S_SIZE, a_size=A_SIZE)
    actor.set_theta(theta)

    state, _ = env.reset()
    img = plt.imshow(env.render())
    rewards = []
    while True:
        action = actor.act(state)
        img.set_data(env.render())
        plt.axis("off")
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
        plt.pause(0.1)

    env.close()


if __name__ == "__main__":
    S_SIZE = 4
    A_SIZE = 2
    T_MAX = 1000
    N_EPS = 2

    bounds = [(0, 1) for _ in range(S_SIZE * A_SIZE)]  # boundaries of theta

    env = gym.make("CartPole-v1")
    actor = Actor(s_size=S_SIZE, a_size=A_SIZE)

    start_time = timeit.default_timer()
    result = shgo(
        fun,
        bounds,
        n=int(1e4),
        workers=1,
        sampling_method="simplicial",
    )
    print(f"Took {timeit.default_timer() - start_time:.2f} seconds.")
    env.close()

    evaluate(result.x)

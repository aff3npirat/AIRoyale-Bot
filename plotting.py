import numpy as np
import matplotlib.pyplot as plt

from memory import MemoryIterator
from constants import KING_HP, PRINCESS_HP



def tail_distribution(*files, output=None, max_reward=101, grid_size=20, linestyle="x-", alpha=0.8):
    tail_dists = np.empty((grid_size, len(files)))
    
    for i, path in enumerate(files):
        rewards = []

        with MemoryIterator(path, ["reward", "done"]) as memory:
            episode_reward = 0.0
            for reward, done in memory:
                episode_reward += reward

                if done:
                    rewards.append(episode_reward)
                    episode_reward = 0.0

        rewards = np.array(rewards, dtype="f4")
        grid = np.linspace(0.0, max_reward, grid_size)
        tail_dists[:, i] = np.mean(rewards.reshape(1, -1)>=grid.reshape(grid_size, 1), axis=1, dtype="f4")

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(20, 20))
    ax.plot(grid_size, tail_dists, linestyle, alpha=alpha)
    ax.set(
        ylim=(0.0, max_reward),
        ylabel="Avg. runs with 'reward > r'",
        xticks=(grid),
        xlabel="Reward 'r'",
    )

    if output is None:
        fig.show()
    else:
        fig.savefig(output)

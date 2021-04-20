import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from mountaincar import MountainCar


def animate_track() -> None:

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlim=MountainCar.position_bound, ylim=(-1.1, 1.1))

    plt.title('Track')
    plt.ylabel('y')
    plt.xlabel('x')

    x = np.linspace(MountainCar.position_bound[0], MountainCar.position_bound[1], 91)
    t = np.linspace(1, 100, 91)
    y_x = y(x)

    plt.plot(x, y_x)

    point = ax.plot([0], [0], color='green', marker='o')[0]
    point.set_data(0, 0)

    def animate(i):
        point.set_data(x[i], y_x[i])
        return point

    anim = FuncAnimation(fig, animate, interval=100, frames=len(t) - 1)

    plt.tight_layout()
    plt.draw()
    plt.show()
    anim.save('plots/track.gif', writer='imagemagick')
    plt.close()


def plot_steps_per_episode(steps_per_episode: List[int]) -> None:
    plt.title('Steps per episode')
    plt.ylabel('Steps')
    plt.xlabel('Episode')
    episodes = np.linspace(1, len(steps_per_episode) + 1, len(steps_per_episode))
    plt.scatter(episodes, steps_per_episode)

    plt.savefig('plots/steps_per_episode.png')
    plt.close()


def plot_loss(loss_history: List[int]) -> None:
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Episode')
    plt.plot(loss_history, label="(training) loss")
    plt.legend()
    plt.savefig('plots/loss.png')
    plt.close()


def plot_epsilon(epsilon_history) -> None:
    plt.title('Epsilon')
    plt.xlabel('Time step')
    plt.ylabel('$\epsilon$')

    x = [i for i in range(len(epsilon_history))]
    explore_history = [y for y in epsilon_history if y > 0.5]
    exploit_history = [y for y in epsilon_history if y <= 0.5]

    plt.plot(x[:len(explore_history)], explore_history, label='Explorative', color='tab:cyan')
    plt.plot(x[len(explore_history):], exploit_history, label='Exploitative', color='tab:blue')

    plt.legend()
    plt.savefig('plots/epsilon.png')
    plt.close()


def plot_track() -> None:
    plt.title('Track')
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.linspace(MountainCar.position_bound[0], MountainCar.position_bound[1], 91)

    plt.plot(x, y(x))

    plt.tight_layout()
    plt.savefig('plots/track.png')
    plt.close()


def y(x):
    return np.cos(3 * (x + math.pi / 2))


if __name__ == "__main__":
    animate_track()
    plot_track()

# type: ignore
""" visualize.py
    Utility module for plotting with matplotlib
"""

import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import ArrowStyle, FancyArrowPatch

from MountainCar import Action, MountainCar
import parameters


def animate_track(state_action_history: List[Tuple[float, Action]], filename: Optional[str] = None) -> None:
    state_history, action_history = zip(*state_action_history)
    state_history, action_history = np.array(state_history), np.array(action_history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlim=MountainCar.position_bound, ylim=(-1.1, 1.1))

    plt.title('Track')
    plt.ylabel('y')
    plt.xlabel('x')

    x = np.linspace(MountainCar.position_bound[0], MountainCar.position_bound[1], 91)
    plt.plot(x, y(x))

    y_t = y(state_history)

    point = ax.plot(state_history[0], y_t[0], color='green', marker='o')[0]
    a = ax.add_patch(FancyArrowPatch((state_history[0], y_t[0]), (state_history[0] + 0.1 * action_history[0], y_t[0])))  # noqa

    point.set_data(0, 0)

    def animate(i):
        ax.patches.pop(0)
        point.set_data(state_history[i], y_t[i])
        a = ax.add_patch(FancyArrowPatch((state_history[i], y_t[i]), (state_history[i] + 0.1 * action_history[i], y_t[i]), color='green', arrowstyle=ArrowStyle("->", head_length=1.5, head_width=1.5)))  # noqa
        return point

    anim = FuncAnimation(fig, animate, interval=100, frames=len(state_history) - 1)

    plt.tight_layout()
    plt.draw()

    if not filename:
        plt.show()
    else:
        anim.save(f'models/{filename}.gif', writer='pillow')
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
    plt.ylabel('$\epsilon$')  # noqa

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


def visualize_tilings(tilings=None) -> None:
    # Lower bounds for each dimension of the continuous space --> (x, y).
    low = [parameters.X_RANGE[0], parameters.Y_RANGE[0]]
    # Upper bounds for each dimension of the continuous space --> (x, y).
    high = [parameters.X_RANGE[1], parameters.Y_RANGE[1]]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        line = None
        for x in grid[0]:
            line = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            line = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(line)
    ax.grid(False)
    ax.legend(legend_lines, ["Tiling #{}".format(t)
              for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
    ax.set_title("Tilings")
    plt.xlim(low[0], high[0])
    plt.ylim(low[1], high[1])
    plt.savefig('plots/tiles.png')
    plt.close()

import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import parameters


def plot_loss(loss_history: List[int]):
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Episode')
    plt.plot(loss_history, label="(training) loss")
    plt.legend()
    plt.savefig('plots/loss.png')
    plt.close()


def plot_epsilon(epsilon_history):
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


def plot_track():
    def y(x):
        return np.cos(3 * (x + math.pi / 2))

    plt.title('Track')
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.arange(-1.2, 0.6, 0.01)

    plt.plot(x, y(x))

    plt.tight_layout()
    plt.savefig('plots/track.png')
    plt.close()


if __name__ == "__main__":
    plot_track()

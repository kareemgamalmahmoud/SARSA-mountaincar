import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import parameters
from mountaincar import MountainCar

# Global config
plt.style.use('ggplot')


def animate_track():

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlim=MountainCar.position_bound, ylim=(-1.1, 1.1))
    ax.set_facecolor('white')

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
    plt.title('Track')
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.arange(-1.2, 0.6, 0.01)

    plt.plot(x, y(x))

    plt.tight_layout()
    plt.savefig('plots/track.png')
    plt.close()


def y(x):
    return np.cos(3 * (x + math.pi / 2))


if __name__ == "__main__":
    animate_track()

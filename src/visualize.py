from typing import List

import matplotlib.pyplot as plt

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


def plot_win_statistics(statistics):
    plt.title('TOPP results')
    plt.xlabel('Agent')
    plt.xticks(rotation=45)
    plt.ylabel('wins')
    agents = statistics.keys()
    wins = statistics.values()
    plt.bar(agents, wins)

    plt.tight_layout()
    plt.savefig('plots/TOPP_results.png')
    plt.close()

from os import walk

from ANET import ANET

import parameters
from reinforcement_learner import ReinforcementLearner
from visualize import Visualize


class TOPP:

    def __init__(self) -> None:
        self.agents = TOPP.get_agents()
        self.number_of_agents = len(self.agents)
        self.number_of_games = parameters.NUMBER_OF_GAMES
        self.visualize_game = parameters.VISUALIZE_GAMES
        self.number_of_games_per_agent = self.number_of_games * (self.number_of_agents - 1)

    @staticmethod
    def get_agents():
        _, _, models = next(walk('models'))
        models = filter(lambda name: name[0] != '.', models)
        models = sorted(models, key=lambda name: int(name.split(".")[0]))
        agents = []
        for model in models:
            agent = ANET(model)
            agents.append(agent)
        return agents

    def run(self):
        win_statistics = {str(agent): 0 for agent in self.agents}
        for i in range(self.number_of_agents - 1):
            player_1 = self.agents[i]
            for j in range(i + 1, self.number_of_agents):
                player_2 = self.agents[j]

                for game in range(self.number_of_games):
                    print(f'p1={player_1} is playing against p2={player_2}. Round {game + 1}')
                    winner = ReinforcementLearner.run_one_game(player_1, player_2, self.visualize_game)
                    winner = str(player_1) if winner == 1 else str(player_2)
                    win_statistics[winner] += 1
                    player_1, player_2 = player_2, player_1  # change the starting player

        self.plot_win_statistics(win_statistics)

    def plot_win_statistics(self, statisitcs):
        Visualize.plot_win_statistics(statisitcs)
        for agent, wins in statisitcs.items():
            print(f'{agent:>10} won {wins}/{self.number_of_games_per_agent} games')

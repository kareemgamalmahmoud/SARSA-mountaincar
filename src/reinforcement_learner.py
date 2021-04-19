import random
from time import time
from typing import Tuple

import numpy as np

import parameters
from ANET import ANET
from game import Game
from MCTS import MCTS
from visualize import Visualize
from world.simulated_world_factory import SimulatedWorldFactory


class ReinforcementLearner:
    """
    Reinforcement Learner agent using the Actor-Critic architecture

    ...

    Attributes
    ----------

    Methods
    -------
    run() -> None:
        Runs all episodes with pivotal parameters
    run_one_game(player_1: ANET, player_2: ANET, visualize=False) -> None:
        Runs excatly one game with the provided players.
    """

    def __init__(self) -> None:
        self.__actual_game = SimulatedWorldFactory.get_simulated_world()
        self.__replay_buffer = np.empty((0, parameters.STATE_SIZE + parameters.NUMBER_OF_ACTIONS))  # RBUF
        self.__ANET = ANET()

        self.__episodes = parameters.EPISODES
        self.__min_number_of_roullouts = parameters.MIN_NUMBER_OF_ROLLOUTS
        self.__simulation_time_out = parameters.SIMULATION_TIME_OUT
        self.__caching_interval = self.__episodes // (parameters.ANETS_TO_BE_CACHED - 1)
        self.__batch_size = parameters.ANET_BATCH_SIZE
        self.__replay_buffer_size = parameters.REPLAY_BUFFER_SIZE
        self.__buffer_insertion_index = 0

    def __run_one_episode(self,) -> None:
        initial_game_state = self.__actual_game.reset()
        monte_carlo_tree = MCTS(initial_game_state)
        root_state = initial_game_state

        while not self.__actual_game.is_final_state():
            monte_carlo_game = SimulatedWorldFactory.get_simulated_world(root_state)

            number_of_rollouts = 0
            start_time = time()
            while time() - start_time < self.__simulation_time_out or number_of_rollouts < self.__min_number_of_roullouts:
                monte_carlo_tree.do_one_simulation(self.__ANET.choose_epsilon_greedy, monte_carlo_game)
                monte_carlo_game.reset(root_state)
                number_of_rollouts += 1
            # print(f'Rollouts: {number_of_rollouts}')

            target_distribution = monte_carlo_tree.get_normalized_distribution()
            self.__add_to_replay_buffer(root_state, target_distribution)

            action = monte_carlo_tree.root.tree_policy()
            next_state, _ = self.__actual_game.step(action)

            monte_carlo_tree.update_root(action)
            root_state = next_state

        # Train ANET on a random minibatch of cases from RBUF
        random_rows = self.__sample_replay_buffer()
        self.__ANET.fit(self.__replay_buffer[random_rows])

    def __add_to_replay_buffer(self, root_state: Tuple[int, ...], target_distribution: Tuple[float, ...]):
        training_instance = np.array([root_state + target_distribution], dtype=np.float64)
        if self.__buffer_insertion_index < self.__replay_buffer_size:
            self.__replay_buffer = np.append(self.__replay_buffer, training_instance, axis=0)  # type: ignore
        else:
            i = self.__buffer_insertion_index % self.__replay_buffer_size
            self.__replay_buffer[i] = training_instance  # type: ignore
        self.__buffer_insertion_index += 1

    def __sample_replay_buffer(self):
        number_of_rows = min(self.__buffer_insertion_index, self.__replay_buffer_size)
        batch_size = min(number_of_rows, self.__batch_size)
        return random.sample(range(0, number_of_rows), batch_size)

    def run(self) -> None:
        """
        Runs all episodes with pivotal parameters.
        Visualizes one round at the end.
        """
        self.__ANET.save('0.h5')  # Save the untrained ANET prior to episode 1
        for episode in range(1, self.__episodes + 1):
            print('\nEpisode:', episode)
            self.__run_one_episode()

            if episode % self.__caching_interval == 0:
                # Save ANET for later use in tournament play.
                self.__ANET.save(str(episode) + '.h5')

        Visualize.plot_loss(self.__ANET.loss_history)
        Visualize.plot_epsilon(self.__ANET.epsilon_history)

        if parameters.VISUALIZE_GAMES:
            print('Showing one episode with the greedy strategy.')
            ReinforcementLearner.run_one_game(self.__ANET, self.__ANET, True)

    @staticmethod
    def run_one_game(player_1: ANET, player_2: ANET, visualize: bool) -> int:
        """
        Runs excatly one game with the provided players.
        """
        world = SimulatedWorldFactory.get_simulated_world()
        current_state = world.reset()

        if visualize and parameters.GAME_TYPE == Game.Hex:
            Visualize.initialize_board(current_state)

        players = (player_1, player_2)
        i = 0
        winner = 0
        while not world.is_final_state():
            legal_actions = world.get_legal_actions()

            action = players[i].choose_greedy(current_state, legal_actions)
            current_state, winner = world.step(action)

            # Alternating players
            i = (i + 1) % 2

            if visualize and parameters.GAME_TYPE == Game.Hex:
                Visualize.draw_board(current_state, winner, str(player_1), str(player_2))

        print(f'Player {winner} won the game.')
        return winner

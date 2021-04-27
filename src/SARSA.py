from typing import List, Tuple

import parameters
import visualize
from Agent import Agent
from MountainCar import Action, MountainCar


class SARSA:

    def __init__(self):
        self.episodes = parameters.EPISODES
        self.caching_interval = parameters.CACHING_INTERVAL
        self.agent = Agent()
        self.simulated_world = MountainCar()
        self.steps_per_episode = []

    def run_one_episode(self, max_steps: int = 1000) -> Tuple[int, List[Tuple[float, Action]]]:
        self.agent.decay_epsilon()
        self.agent.reset_eligibilities()
        state, reward, done = self.simulated_world.reset()
        action = self.agent.choose_epsilon_greedy(state)

        done = False
        steps = 0
        state_action_history = [(state[0], action)]
        while not done and steps < max_steps:
            next_state, reward, done = self.simulated_world.step(action)
            next_action = self.agent.choose_epsilon_greedy(next_state)

            self.agent.update(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
            steps += 1
            state_action_history.append((state[0], action))
        return steps, state_action_history

    def run(self) -> None:
        """
        Runs all episodes with pivotal parameters.
        Visualizes one round at the end.
        """
        for episode in range(1, self.episodes + 1):
            print('Episode:', episode)
            steps, state_action_history = self.run_one_episode()
            self.steps_per_episode.append(steps)
            if episode % parameters.CACHING_INTERVAL == 0:
                visualize.animate_track(state_action_history, f'agent-{episode}')

        print('Training completed.')
        visualize.plot_steps_per_episode(self.steps_per_episode)
        visualize.plot_epsilon(self.agent.epsilon_history)

        if parameters.VISUALIZE_FINAL_GAME:
            print('Showing one episode with the greedy strategy.')
            self.agent.epsilon = 0
            steps, state_action_history = self.run_one_episode()
            print(f'Episode completed in {steps} steps.')
            visualize.animate_track(state_action_history)

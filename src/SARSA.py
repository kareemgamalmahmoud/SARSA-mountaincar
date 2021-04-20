import parameters
from mountaincar import MountainCar
from Agent import Agent


class SARSA:

    def __init__(self):
        self.episodes = parameters.EPISODES
        self.agent = Agent()
        self.simulated_world = MountainCar()
        self.steps_per_episode = []

    def run_one_episode(self, max_steps: int = 1000) -> int:
        self.agent.reset_eligibilities()
        state, reward, done = self.simulated_world.reset()
        action = self.agent.choose_epsilon_greedy(state)

        done = False
        steps = 0
        while not done or steps >= max_steps:
            next_state, reward, done = self.simulated_world.step(action)
            next_action = self.agent.choose_epsilon_greedy(next_state)

            self.agent.update(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
            steps += 1
        return steps

    def run(self) -> None:
        """
        Runs all episodes with pivotal parameters.
        Visualizes one round at the end.
        """
        for episode in range(self.episodes):
            print('Episode:', episode + 1)
            steps = self.run_one_episode()
            self.steps_per_episode.append(steps)

        print('Training completed.')
        # Plotting ...

        if parameters.VISUALIZE_GAMES:
            print('Showing one episode with the greedy strategy.')
            self.SARSA.epsilon = 0
            self.run_one_episode()

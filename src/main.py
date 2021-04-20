import glob
import os

import parameters
from SARSA import SARSA
from TOPP import TOPP
from Agent import Agent


def clear_models() -> None:
    files = glob.glob('models/*')
    for f in files:
        os.remove(f)


def load_agent() -> Agent:
    return Agent("models/" + input("Modelnavn: ") + ".h5")


if __name__ == "__main__":
    if parameters.RUN_TRAINING:
        clear_models()
        rl_learner = SARSA()
        rl_learner.run()
    else:
        agent = load_agent()
        # TODO: Plotting...

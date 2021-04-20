import glob
import os

import parameters
from SARSA import SARSA
from TOPP import TOPP


def clear_models():
    files = glob.glob('models/*')
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    if parameters.RUN_TRAINING:
        clear_models()
        rl_learner = SARSA()
        rl_learner.run()

    topp = TOPP()
    topp.run()

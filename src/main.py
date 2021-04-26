import glob
import os

from SARSA import SARSA


def clear_models() -> None:
    files = glob.glob('models/*')
    for f in files:
        os.remove(f)


if __name__ == "__main__":
    clear_models()

    rl_learner = SARSA()
    rl_learner.run()

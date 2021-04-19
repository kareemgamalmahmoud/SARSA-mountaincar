from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.losses import kl_divergence, mse  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

from game import Game

VISUALIZE_GAMES = False
FRAME_DELAY = 0.5
RUN_TRAINING = False

# RL parameters
EPISODES = 120
REPLAY_BUFFER_SIZE = 512

# MCTS parameters
MIN_NUMBER_OF_ROLLOUTS = 100
SIMULATION_TIME_OUT = 0.0  # s
UCT_C = 1  # "theoretically 1"

# Simulated World
GAME_TYPE = Game.Hex
LEDGE_BOARD = (0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 1, 1)  # (0, 2, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1)
SIZE = 6 if GAME_TYPE == Game.Hex else len(LEDGE_BOARD)  # 3 <= k <= 10
STATE_SIZE = 1 + (SIZE ** 2 if GAME_TYPE == Game.Hex else SIZE)
NUMBER_OF_ACTIONS = SIZE ** 2 if GAME_TYPE == Game.Hex else int((SIZE ** 2 - SIZE) / 2) + 1

# ANET
ANET_EPSILON = 0.01
ANET_EPSILON_DECAY = 1
ANET_LEARNING_RATE = 0.01
ANET_LOSS_FUNCTION = mse  # deepnet_cross_entropy, kl_divergence
ANET_ACTIVATION_FUNCTION = relu  # linear, relu, sigmoid, or tanh
ANET_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
ANET_DIMENSIONS = (STATE_SIZE, 256, 128, 64, NUMBER_OF_ACTIONS)
ANET_BATCH_SIZE = 64

# TOPP parameters
ANETS_TO_BE_CACHED = 6
NUMBER_OF_GAMES = 10

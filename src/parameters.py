from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.losses import mse  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

VISUALIZE_GAMES = False
FRAME_DELAY = 0.5
RUN_TRAINING = False

# RL parameters
EPISODES = 10
MODELS_TO_BE_CACHED = 10

# Domain & TileEncoder parameters
SIZE_OF_TILING_X = 4
SIZE_OF_TILING_Y = 4
NUM_OF_TILINGS = 4
NUMBER_OF_FEATURES = SIZE_OF_TILING_X * SIZE_OF_TILING_Y * NUM_OF_TILINGS

X_RANGE = (-1.2, 0.6)
Y_RANGE = (-0.07, 0.07)

# SARSA
SARSA_EPSILON = 0.01
SARSA_EPSILON_DECAY = 1
SARSA_DISCOUNT_FACTOR = 0.9
SARSA_TRACE_DECAY = 0.9

SARSA_LEARNING_RATE = 0.01
SARSA_LOSS_FUNCTION = mse  # "A typical solution involves mean squared error as the loss function for your critic network"
SARSA_ACTIVATION_FUNCTION = linear  # linear, relu, sigmoid, or tanh
SARSA_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
SARSA_DIMENSIONS = (NUMBER_OF_FEATURES, 1)

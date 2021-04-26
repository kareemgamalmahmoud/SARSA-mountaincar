from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.losses import mse  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

VISUALIZE_GAMES = True
FRAME_DELAY = 0.5
RUN_TRAINING = False

# RL parameters
EPISODES = 100
CACHING_INTERVAL = 10

# Domain & TileEncoder parameters
SIZE_OF_TILING_X = 4
SIZE_OF_TILING_Y = 4
NUM_OF_TILINGS = 4
NUMBER_OF_FEATURES = SIZE_OF_TILING_X * SIZE_OF_TILING_Y * NUM_OF_TILINGS

X_RANGE = (-1.2, 0.6)
Y_RANGE = (-0.07, 0.07)

# Agent
EPSILON = 1
EPSILON_DECAY = 0.96
DISCOUNT_FACTOR = 0.92
TRACE_DECAY = 0.9

NN_LEARNING_RATE = 0.001
NN_DIMENSIONS = (NUMBER_OF_FEATURES, 1)
NN_ACTIVATION_FUNCTION = relu  # linear, relu, sigmoid, or tanh
NN_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
NN_LOSS_FUNCTION = mse  # "A typical solution involves mean squared error as the loss function for your critic network"

from keras.activations import linear, relu, sigmoid, tanh  # noqa
from keras.losses import kl_divergence, mse  # noqa
from keras.optimizers import SGD, Adagrad, Adam, RMSprop  # noqa

VISUALIZE_GAMES = False
FRAME_DELAY = 0.5
RUN_TRAINING = False

# RL parameters
EPISODES = 120

# ANET
ANET_EPSILON = 0.01
ANET_EPSILON_DECAY = 1
ANET_LEARNING_RATE = 0.01
ANET_LOSS_FUNCTION = mse  # deepnet_cross_entropy, kl_divergence
ANET_ACTIVATION_FUNCTION = relu  # linear, relu, sigmoid, or tanh
ANET_OPTIMIZER = Adam  # SGD, Adagrad, Adam, or RMSprop
# ANET_DIMENSIONS = (STATE_SIZE, 256, 128, 64, NUMBER_OF_ACTIONS)

# TOPP parameters
ANETS_TO_BE_CACHED = 6
NUMBER_OF_GAMES = 10

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K  # noqa
from keras.activations import softmax
from keras.layers import Dense, Input
from keras.models import Sequential

import parameters


class ANET:
    """
    Actor NETwork using the epsilon-greedy strategy

    ...

    Attributes
    ----------
    loss_history
    epsilon_history

    Methods
    -------
    save(model_name: str) -> None:
        Saves model with model_name to t 'models/' for later use in TOPP play.
    load(self, model_name: str, directory: str) -> None:
        Loads model with model_name from 'models/' for use in TOPP play.
    choose_uniform(valid_actions: Tuple[int, ...]) -> int:
        Chooses a random action from valid_actions
    choose_greedy(state: Tuple[int, ...], valid_actions: Tuple[int, ...]) -> int:
        Chooses an action with the greatest likelihood renormalized based on valid_actions
    choose_action(state: Tuple[int, ...], valid_actions: Tuple[int, ...]) -> int:
        Choses an action uniformly with a probability of epsilon otherwise greedily with a probability of (1 - epsilon)
    fit(batch: np.ndarray) -> None:
        Trains the model on the dataset supervised learning style.
    """

    def __init__(self, model_name: Optional[str] = None, directory: str = 'models') -> None:
        self.__epsilon = parameters.ANET_EPSILON
        self.__epsilon_decay_rate = parameters.ANET_EPSILON_DECAY

        if model_name is None:
            self.__loss_function = parameters.ANET_LOSS_FUNCTION
            self.__learning_rate = parameters.ANET_LEARNING_RATE
            self.__activation_function = parameters.ANET_ACTIVATION_FUNCTION
            self.__optimizer = parameters.ANET_OPTIMIZER

            self.__model: Sequential = self.__build_model()
        else:
            self.load(model_name, directory)

        self.__loss_history = []
        self.__epsilon_history = []

    def __build_model(self) -> Sequential:
        """
        Builds a neural network model with the provided dimensions and learning rate
        """
        self.__name = 'Training model'
        input_dim, *hidden_dims, output_dim = parameters.ANET_DIMENSIONS

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation=self.__activation_function))

        model.add(Dense(output_dim, activation=softmax))

        model.compile(
            optimizer=(self.__optimizer(learning_rate=self.__learning_rate) if self.__learning_rate is not None else self.__optimizer()),
            loss=self.__loss_function
        )
        model.summary()
        return model

    @property
    def loss_history(self):
        return self.__loss_history

    @property
    def epsilon_history(self):
        return self.__epsilon_history

    def save(self, model_name: str) -> None:
        self.__model.save(f'models/{model_name}')

    def load(self, model_name: str, directory: str) -> None:
        self.__name = 'Agent-e' + model_name.replace('.h5', '')
        self.__model = tf.keras.models.load_model(f'{directory}/{model_name}', compile=False)  # type: ignore

    def choose_epsilon_greedy(self, state: Tuple[int, ...], valid_actions: Tuple[int, ...]) -> int:
        """Epsilon-greedy action selection function."""
        if random.random() < self.__epsilon:
            return self.choose_uniform(valid_actions)
        return self.choose_greedy(state, valid_actions)

    def choose_uniform(self, valid_actions: Tuple[int, ...]) -> int:
        assert sum(valid_actions) > 0, 'Illegal argument, valid actions cannot be empty'
        return random.choice([i for i, action in enumerate(valid_actions) if action == 1])

    def choose_greedy(self, state: Tuple[int, ...], valid_actions: Tuple[int, ...]) -> int:
        action_probabilities = self.__model(tf.convert_to_tensor([state])).numpy()  # type: ignore
        action_probabilities = action_probabilities * np.array(valid_actions)
        action_probabilities = action_probabilities.flatten()
        action_probabilities /= np.sum(action_probabilities)  # normalize probability distribution
        return np.argmax(action_probabilities)

    def choose_softmax(self, state: Tuple[int, ...], valid_actions: Tuple[int, ...], temperature: int) -> int:
        action_probabilities = self.__model(tf.convert_to_tensor([state])).numpy().flatten()  # type: ignore
        action_probabilities = action_probabilities * np.array(valid_actions)
        action_probabilities = softmax_v2(action_probabilities, temperature)
        return np.random.choice(range(0, len(valid_actions)), 1, p=action_probabilities)[0]

    def fit(self, batch: np.ndarray) -> None:
        X, Y = batch[:, :parameters.STATE_SIZE], batch[:, parameters.STATE_SIZE:]
        history = self.__model.fit(X, Y, batch_size=parameters.ANET_BATCH_SIZE)

        # Used for visualization
        self.__loss_history.append(history.history["loss"][0])  # type: ignore
        self.__epsilon_history.append(self.__epsilon)

        self.__epsilon *= self.__epsilon_decay_rate  # decay epsilon

    def __str__(self) -> str:
        return self.__name

    def __repr__(self) -> str:
        return self.__name


def softmax_v2(x, temperature=1):
    return np.exp(x / temperature) / sum(np.exp(x / temperature))

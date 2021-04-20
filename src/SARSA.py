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
from src.mountaincar import Actions


class SARSA:
    """
    SARSA using the epsilon-greedy strategy

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

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.epsilon = parameters.SARSA_EPSILON
        self.epsilon_decay_rate = parameters.SARSA_EPSILON_DECAY
        self.discount_factor = parameters.SARSA_DISCOUNT_FACTOR
        self.trace_decay = parameters.SARSA_TRACE_DECAY

        if model_path is None:
            self.learning_rate = parameters.SARSA_LEARNING_RATE
            self.dimensions = parameters.SARSA_DIMENSIONS
            self.activation_function = parameters.SARSA_ACTIVATION_FUNCTION
            self.optimizer = parameters.SARSA_OPTIMIZER
            self.loss_function = parameters.SARSA_LOSS_FUNCTION

            self.Q: Sequential = self.build_model()
        else:
            self.load(model_path)

        self.reset_eligibilities()
        self.epsilon_history = []

    def build_model(self) -> Sequential:
        """
        Builds a neural network model with the provided dimensions and learning rate
        """
        self.name = 'Training model'
        input_dim, *hidden_dims, output_dim = self.dimensions

        assert output_dim == 1, 'Output dimension must be 1'

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation=self.activation_function))

        model.add(Dense(units=1, activation='linear'))

        model.compile(
            optimizer=(self.optimizer(learning_rate=self.learning_rate) if self.learning_rate is not None else self.optimizer()),
            loss=self.loss_function
        )
        model.summary()
        return model

    def save(self, model_path: str) -> None:
        self.Q.save(model_path)

    def load(self, model_path: str) -> None:
        self.name = 'Agent-e' + model_path.replace('.h5', '')
        self.Q = tf.keras.models.load_model(f'{model_path}', compile=False)  # type: ignore

    def choose_epsilon_greedy(self, state: Tuple[int, ...]) -> int:
        """Epsilon-greedy action selection function."""
        if random.random() < self.epsilon:
            return self.choose_uniform()
        return self.choose_greedy(state)

    def choose_uniform(self) -> int:
        raise NotImplementedError

    def choose_greedy(self, state: Tuple[int, ...]) -> Actions:
        action_probabilities = []
        for action in [-1, 0, 1]:
            action_probabilities.append(float(self.Q(tf.convert_to_tensor([state, action]))))  # type: ignore
        return -1

    def choose_softmax(self, state: Tuple[int, ...], temperature: int) -> Actions:
        # action_probabilities = np.array(self.Q(tf.convert_to_tensor([state])))
        # action_probabilities = softmax_v2(action_probabilities, temperature)
        # return np.random.choice(range(0, 10), 1, p=action_probabilities)[0]  # ??
        raise NotImplementedError

    def update(self, reward: float, successor_state: Tuple[int], current_state: Tuple[int]) -> None:
        """Updates eligibilities, then the value function."""

        with tf.GradientTape(persistent=True) as tape:
            target = reward + self._discount_factor * self.Q(tf.convert_to_tensor([successor_state]))  # type: ignore
            prediction = self.Q(tf.convert_to_tensor([current_state]))
            loss = self.Q.compiled_loss(target, prediction)
            td_error = target - prediction

        gradients = tape.gradient(loss, self.Q.trainable_weights)
        gradients = self.__modify_gradients(gradients, td_error)
        self.Q.optimizer.apply_gradients(zip(gradients, self.Q.trainable_weights))  # type: ignore

    def __modify_gradients(self, gradients, td_error):
        for gradient, eligibility in zip(gradients, self.eligibilities):
            gradient *= 1 / (2 * td_error)
            eligibility = self.discount_factor * self.trace_decay * eligibility + gradient
            gradient = td_error * eligibility
        return gradients

    def reset_eligibilities(self) -> None:
        """Sets all eligibilities to 0.0"""
        self.eligibilities = []
        for weights in self.Q.trainable_weights:
            self.eligibilities.append(tf.zeros(weights.shape))

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


def softmax_v2(x: int, temperature: float = 1.0) -> float:
    return np.exp(x / temperature) / sum(np.exp(x / temperature))

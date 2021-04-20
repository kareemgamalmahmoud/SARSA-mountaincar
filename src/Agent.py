from __future__ import annotations

import random
from typing import Optional

import numpy as np
import tensorflow as tf
from keras import backend as K  # noqa
from keras.layers import Dense, Input
from keras.models import Sequential

import parameters
from mountaincar import Action, State


class Agent:

    actions: list[Action] = [-1, 0, 1]

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

    def choose_epsilon_greedy(self, state: State) -> Action:
        """Epsilon-greedy action selection function."""
        if random.random() < self.epsilon:
            return self.choose_uniform()
        return self.choose_greedy(state)

    def choose_uniform(self) -> Action:
        return random.choice(Agent.actions)

    def choose_greedy(self, state: State) -> Action:
        action_probabilities = []
        for action in Agent.actions:
            action_probabilities.append(float(self.Q(tf.convert_to_tensor([state, action]))))  # type: ignore
        return Agent.actions[np.argmax(action_probabilities)]

    def choose_stochastic(self, state: State, temperature: int = 1) -> Action:
        # action_probabilities = np.array(self.Q(tf.convert_to_tensor([state])))
        # action_probabilities = softmax_v2(action_probabilities, temperature)
        # return np.random.choice(range(0, 10), 1, p=action_probabilities)[0]  # ??
        raise NotImplementedError

    def update(self, state: State, action: Action, reward: int, next_state: State, next_action: Action) -> None:
        """Updates eligibilities, then the value function."""

        with tf.GradientTape(persistent=True) as tape:
            target = reward + self._discount_factor * self.Q(tf.convert_to_tensor([next_state, next_action]))  # type: ignore
            prediction = self.Q(tf.convert_to_tensor([state, action]))
            loss = self.Q.compiled_loss(target, prediction)
            td_error = target - prediction

        gradients = tape.gradient(loss, self.Q.trainable_weights)
        gradients = self.modify_gradients(gradients, td_error)
        self.Q.optimizer.apply_gradients(zip(gradients, self.Q.trainable_weights))  # type: ignore

    def modify_gradients(self, gradients, td_error):
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

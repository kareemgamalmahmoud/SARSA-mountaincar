from typing import Tuple

import tensorflow as tf
from keras import backend as K  # noqa
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD

from .critic import Critic


class NNCritic(Critic):
    """
    Neural network based Critic

    ...

    Attributes
    ----------

    Methods
    -------
    update(current_state, successor_state, reward):
        Updates eligibilities, then the value function.
    reset_eligibilities():
        Sets all eligibilities to 0.0
    replace_eligibilities(state, action):
        Not used by NNCritic.
    """

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        trace_decay: float,
        nn_dimensions: tuple
    ):
        super().__init__(
            learning_rate,  # alpha
            discount_factor,  # gamma
            trace_decay,  # lambda
        )
        assert nn_dimensions is not None, 'nn_dimensions cannot be None when using NN-based critic'
        self.__nn_dimensions = nn_dimensions
        self.__values = self.__build_critic_network()  # V(s)
        self.reset_eligibilities()

    def __build_critic_network(self) -> Sequential:
        """Builds a neural network model with the provided dimensions and learning rate"""
        input_dim, *hidden_dims, output_dim = self.__nn_dimensions

        assert output_dim == 1, 'Output dimension must be 1'

        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        for dimension in hidden_dims:
            model.add(Dense(dimension, activation='swish'))

        model.add(Dense(units=1, activation='linear'))

        model.compile(
            optimizer=SGD(learning_rate=self._learning_rate),
            loss='mean_squared_error'
        )
        model.summary()
        return model

    def _get_value(self, state: Tuple[int]) -> float:
        """Value function V(s)"""
        return float(self.__values(tf.convert_to_tensor([state])))  # type: ignore

    def update(self, reward: float, successor_state: Tuple[int], current_state: Tuple[int]) -> None:
        """Updates eligibilities, then the value function."""

        with tf.GradientTape(persistent=True) as tape:
            target = reward + self._discount_factor * self.__values(tf.convert_to_tensor([successor_state]))  # type: ignore
            prediction = self.__values(tf.convert_to_tensor([current_state]))
            loss = self.__values.compiled_loss(target, prediction)
            td_error = target - prediction

        gradients = tape.gradient(loss, self.__values.trainable_weights)
        gradients = self.__modify_gradients(gradients, td_error)
        self.__values.optimizer.apply_gradients(zip(gradients, self.__values.trainable_weights))  # type: ignore

    def __modify_gradients(self, gradients, td_error):
        for gradient, eligibility in zip(gradients, self.__eligibilities):
            gradient *= 1 / (2 * td_error)
            eligibility = self._discount_factor * self._trace_decay * eligibility + gradient
            gradient = td_error * eligibility
        return gradients

    def reset_eligibilities(self) -> None:
        """Sets all eligibilities to 0.0"""
        self.__eligibilities = []
        for weights in self.__values.trainable_weights:
            self.__eligibilities.append(tf.zeros(weights.shape))

    def replace_eligibilities(self, _) -> None:
        """Not used by NNCritic."""
        pass

    def plot_training_data(self):
        """Not used by NNCritic."""
        pass

import random
from math import cos
from typing import Literal, Tuple, Union

Actions = Union[Literal[-1], Literal[0], Literal[1]]
State = Tuple[float, float]
Output = Tuple[State, int, bool]


class MountainCar:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Output:
        self.position = -0.6 + random.random() * 0.2
        self.velocity = 0.0
        self.checkState()
        return (self.position, self.velocity), int(self.is_final_state()), self.is_final_state()

    def step(self, action: Actions) -> Output:
        self.velocity += 0.001 * action - 0.0025 * cos(3 * self.position)
        self.position += self.velocity
        self.checkState()
        return (self.position, self.velocity), int(self.is_final_state()), self.is_final_state()

    def is_final_state(self) -> bool:
        return self.position >= 0.6

    def checkState(self) -> None:
        assert -1.2 <= self.position <= 0.6, f'Position out of bounds. {self.position} not in [-1.2, 0.6]'
        assert -0.07 <= self.velocity <= 0.07, f'Velocity out of bounds. {self.velocity} not in [-0.07, 0.07]'

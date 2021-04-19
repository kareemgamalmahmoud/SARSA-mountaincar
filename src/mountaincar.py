import random
from math import cos
from typing import Literal, Tuple, Union

Actions = Union[Literal[-1], Literal[0], Literal[1]]
State = Tuple[float, float]


class MountainCar:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Tuple[State, int]:
        self.position = -0.6 + random.random() * 0.2
        self.velocity = 0
        return (self.position, self.velocity), bool(self.position >= 0.6)

    def step(self, action: Actions) -> Tuple[State, int]:
        self.velocity += 0.001 * action - 0.0025 * cos(3 * self.position)
        self.position += self.velocity
        return (self.position, self.velocity), int(self.is_final_state())

    def is_final_state(self) -> bool:
        return self.position >= 0.6

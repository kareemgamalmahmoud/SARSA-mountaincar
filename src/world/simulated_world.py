from abc import ABC, abstractmethod
from typing import Optional, Tuple


class SimulatedWorld(ABC):

    @abstractmethod
    def step(self, action: int) -> Tuple[Tuple[int, ...], int]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, state: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def is_final_state(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_winner_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_legal_actions(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def generate_state(self, action: int) -> Tuple[int, ...]:
        raise NotImplementedError

from typing import Tuple
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod


class Env(ABC):
    
    @abstractmethod
    def step(self, action: int) -> Tuple[ArrayLike, int, bool]:
        pass

    @abstractmethod
    def reset(self) -> ArrayLike:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

class Agent(ABC):

    @abstractmethod
    def act(self, state: ArrayLike) -> int:
        pass

    @abstractmethod
    def remember(self, state: ArrayLike, action: int, reward: int, next_state: ArrayLike, done: bool) -> None:
        pass

    @abstractmethod
    def replay(self, batch_size: int) -> None:
        pass
    
    @abstractmethod
    def train(self, env: Env, n_episodes: int, *, batch_size: int = 128) -> None:
        pass
    
    @abstractmethod
    def evaluate(self, env: Env, n_episodes: int = 100) -> float:
        pass
import random
import torch as th
from collections import deque
from typing import NamedTuple, List
from numpy.typing import ArrayLike

from components.abstractions import Memory


class Transition(NamedTuple):
    state: ArrayLike
    action: int
    reward: int
    next_state: ArrayLike
    done: bool

class ReplayMemory(Memory):
    """
    A replay memory buffer used to store and sample experiences for training.

    ### Parameters:
    - capacity (`int`): The maximum capacity of the replay memory buffer.

    ### Methods:
    - push -> `None`: Add a new transition to the replay memory buffer.
    - sample -> `List[Transition]`: Sample a batch of random transitions from the replay memory buffer.
    - save -> `None`: Save the contents of the replay memory buffer to a file at the specified path.
    - load -> `None`: Load the contents of the replay memory buffer from a file at the specified path.
    
    ### Example:
    ```
    >>> memory = ReplayMemory(2000)
    >>> state = env.reset()
    >>> action = agent.act(state)
    >>> next_state, reward, done = env.step(action)
    >>> memory.push(state, action, reward, next_state, done)
    ```
    """

    def __init__(self, capacity: int) -> None:
        self._buffer = deque([], maxlen=capacity)

    def push(self, *transition) -> None:
        """
        Add a new transition to the replay memory buffer.

        ### Parameters:
        - state (`ArrayLike`): The current state of the environment represented as an array-like object.
        - action (`int`): The action taken by the agent represented as an integer.
        - reward (`int`): The reward received after taking the action.
        - next_state (`ArrayLike`): The resulting state after taking the action represented as an array-like object.
        - done (`bool`): A boolean indicating whether the episode has ended after taking the action.

        ### Returns:
        - `None`
        """

        self._buffer.append(Transition(*transition))

    def sample(self, batch_size: int = 1) -> List[Transition]:
        """
        Sample a batch of random transitions from the replay memory buffer.

        ### Parameters:
        - batch_size (`int`): The number of transitions (experiences) to sample from the replay memory buffer.

        ### Returns:
        - `List[Transition]`: A list of sampled transitions from the replay memory buffer.
        """
        
        return random.sample(self._buffer, batch_size)

    def save(self, path: str) -> None:
        """
        Save the contents of the replay memory buffer to a file at the specified path.

        ### Parameters:
        - path (`str`): The file path where the contents of the replay memory buffer will be saved.

        ### Returns:
        - `None`
        """

        th.save(self._buffer, path)

    def load(self, path: str) -> None:
        """
        Load the contents of the replay memory buffer from a file at the specified path.

        ### Parameters:
        - path (`str`): The file path from where the contents of the replay memory buffer will be loaded.
            
        ### Returns:
        - `None`
        """

        self._buffer.extend(th.load(path))

    def __len__(self) -> int:
        return len(self._buffer)

class PERMemory(Memory):
    pass

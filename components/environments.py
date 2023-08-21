import os
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from typing import Tuple
from numpy.typing import ArrayLike

from components.abstractions import Env, Agent


class CartPoleEnv(Env):
    def __init__(self) -> None:
        self._env = gym.make("CartPole-v1")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self.n_actions = self.action_space.n
        self.n_observations = self.observation_space.shape[0]

    def reset(self) -> ArrayLike:
        state, _ = self._env.reset()
        return state

    def step(self, action: int) -> Tuple[ArrayLike, int, bool]:
        observation, reward, terminated, truncated, _ = self._env.step(action)
        return observation, int(reward), terminated or truncated
    
    def render(self) -> None:
        self._env.render()

class MountainCarEnv(Env):
    def __init__(self) -> None:
        self._env = gym.make("MountainCar-v0")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self.n_actions = self.action_space.n
        self.n_observations = self.observation_space.shape[0]

    def reset(self) -> ArrayLike:
        state, _ = self._env.reset()
        return state

    def step(self, action: int) -> Tuple[ArrayLike, int, bool]:
        observation, reward, terminated, truncated, _ = self._env.step(action)
        return observation, int(reward), terminated or truncated
    
    def render(self) -> None:
        self._env.render()

class TicTacToeEnv(Env):
    def __init__(self, *, opponent: Agent = None, random_action_chance: float = 0.5) -> None:
        self.action_space = Discrete(9)
        self.observation_space = Box(-1, 1, (9,), dtype=np.int8)

        self.n_actions = self.action_space.n
        self.n_observations = self.observation_space.shape[0]

        self._opponent = opponent
        self._random_action_chance = random_action_chance

        self.reset()

    def reset(self) -> ArrayLike:
        self._board = np.zeros((3, 3), dtype=np.float32)
        self._winner = None
        self._done = False

        # 50% chance to take the 2nd turn
        if random.random() > 0.5:
            self._opponent_act()

        return self._get_state()

    def step(self, action: int) -> Tuple[ArrayLike, int, bool]:
        row = action // 3
        col = action % 3

        # Invalid action check (immediate loss)
        if self._board[row, col] != 0:
            self._winner = -1
            self._done = True
            return self._get_state(), self._get_reward(), self._done
        
        self._board[row, col] = 1
        self._winner = self._get_winner()
        self._done = bool(self._is_full() or self._winner)

        if not self._done:
            self._opponent_act()

        return self._get_state(), self._get_reward(), self._done

    def render(self) -> None:
        os.system("cls")
        symbols = {0: "-", 1: "x", -1: "o"}
        for row in self._board:
            print(" | ".join([symbols[s] for s in row]))
            print("-" * 9)

    def set_opponent(self, opponent: Agent) -> None:
        self._opponent = opponent

    def _opponent_act(self) -> None:
        available_actions = [action for action in range(self.action_space.n) if self._board.flatten()[action] == 0]

        if self._opponent and random.random() > self._random_action_chance:
            state = self._get_state() * -1
            action = self._opponent.act(state)
            action = action if action in available_actions else random.choice(available_actions)
        else:
            action = random.choice(available_actions)

        row = action // 3
        col = action % 3

        self._board[row, col] = -1
        self._winner = self._get_winner()
        self._done = bool(self._is_full() or self._winner)

    def _get_winner(self) -> int:
        # Check rows
        for row in self._board:
            if np.all(row == 1):
                return 1
            if np.all(row == -1):
                return -1

        # Check columns
        for col in self._board.T:
            if np.all(col == 1):
                return 1
            if np.all(col == -1):
                return -1
            
        # Check diagonals
        if np.all(np.diag(self._board) == 1) or np.all(np.diag(np.fliplr(self._board)) == 1):
            return 1
        if np.all(np.diag(self._board) == -1) or np.all(np.diag(np.fliplr(self._board)) == -1):
            return -1

    def _is_full(self) -> bool:
        return bool(np.count_nonzero(self._board) == 9)

    def _get_state(self) -> ArrayLike:
        return self._board.flatten()

    def _get_reward(self) -> int:
        if self._winner:
            return self._winner * 10
        elif self._is_full():
            return 5
        return 0

import os
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from typing import Tuple
from numpy.typing import ArrayLike

from components.abstractions import Env, Agent


class CartPoleEnv(Env):
    """
    Modified `CartPole-v1` environment with a negative reward of `-10` on termination.
    """
    def __init__(self, max_steps: int = 500) -> None:
        self.env = gym.make("CartPole-v1", max_episode_steps=max_steps)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.n_actions = self.action_space.n
        self.n_observations = self.observation_space.size[0]

    def reset(self) -> ArrayLike:
        state, _ = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[ArrayLike, int, bool]:
        observation, reward, terminated, truncated, _ = self.env.step(action)
        reward = -10 if terminated else reward

        return observation, int(reward), terminated or truncated
    
    def render(self) -> None:
        self.env.render()

class TicTacToeEnv(Env):
    def __init__(self, *, opponent: Agent = None, random_action_chance: float = 0.5) -> None:
        self.action_space = Discrete(9)
        self.observation_space = Box(-1, 1, (9,), dtype=np.int8)

        self.n_actions = self.action_space.n
        self.n_observations = self.observation_space.size[0]

        self.opponent = opponent
        self.random_action_chance = random_action_chance

        self.reset()

    def reset(self) -> ArrayLike:
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.winner = None
        self.done = False

        # 50% chance to take the 2nd turn
        if random.random() > 0.5:
            self.opponent_act()

        return self.get_state()

    def step(self, action: int) -> Tuple[ArrayLike, int, bool]:
        row = action // 3
        col = action % 3

        # Invalid action check (immediate loss)
        if self.board[row, col] != 0:
            self.winner = -1
            self.done = True
            return self.get_state(), self.get_reward(), self.done
        
        self.board[row, col] = 1
        self.winner = self.get_winner()
        self.done = bool(self.is_full() or self.winner)

        if not self.done:
            self.opponent_act()

        return self.get_state(), self.get_reward(), self.done

    def render(self) -> None:
        os.system("cls")
        symbols = {0: "-", 1: "x", -1: "o"}
        for row in self.board:
            print(" | ".join([symbols[s] for s in row]))
            print("-" * 9)

    def opponent_act(self) -> None:
        available_actions = [action for action in range(self.action_space.n) if self.board.flatten()[action] == 0]

        if self.opponent and random.random() > self.random_action_chance:
            state = self.get_state() * -1
            action = self.opponent.act(state)
            action = action if action in available_actions else np.random.choice(available_actions)
        else:
            action = random.choice(available_actions)

        row = action // 3
        col = action % 3

        self.board[row, col] = -1
        self.winner = self.get_winner()
        self.done = bool(self.is_full() or self.winner)

    def get_winner(self) -> int:
        # Check rows
        for row in self.board:
            if np.all(row == 1):
                return 1
            if np.all(row == -1):
                return -1

        # Check columns
        for col in self.board.T:
            if np.all(col == 1):
                return 1
            if np.all(col == -1):
                return -1
            
        # Check diagonals
        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            return 1
        if np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1):
            return -1

    def is_full(self) -> bool:
        return bool(np.count_nonzero(self.board) == 9)

    def get_state(self) -> ArrayLike:
        return self.board.flatten()

    def get_reward(self) -> int:
        if self.winner:
            return self.winner * 10
        elif self.is_full():
            return 5
        return 0

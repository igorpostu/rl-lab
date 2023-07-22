import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from collections import deque
from copy import deepcopy
from typing import NamedTuple, Any, List
from numpy.typing import ArrayLike
from tqdm import tqdm

from components.abstractions import Agent, Env


class Transition(NamedTuple):
    state: ArrayLike
    action: int
    reward: int
    next_state: ArrayLike
    done: bool

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int = 1) -> List[Transition]:
        return random.sample(self.memory, batch_size)
    
    def save(self, path: str) -> None:
        torch.save(self.memory, path)
    
    def load(self, path: str) -> None:
        self.memory.extend(torch.load(path))

    def __len__(self) -> int:
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, model: nn.Sequential, criterion: Any, optimiser: Any) -> None:
        super(DQN, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser

    def update(self, x: Tensor, y: Tensor) -> None:
        self.train()
        self.optimiser.zero_grad()
        loss = self.criterion(self(x), y)
        loss.backward()
        self.optimiser.step()
        self.eval()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class DQNAgent(Agent):
    def __init__(
            self,
            dqn: DQN,
            memory: ReplayMemory = ReplayMemory(2000),
            *,
            gamma: float = 0.95,
            eps: float = 1,
            eps_decay: float = 0.995,
            eps_min: float = 0.05,
    ) -> None:

        self.dqn = dqn
        self.memory = memory

        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def act(self, state: ArrayLike) -> int:
        # Apply epsilon (random action chance)
        if random.random() <= self.eps:
            n_actions = self.dqn.model[-1].out_features
            return random.randrange(n_actions)

        # Choose action
        with torch.no_grad():
            state = torch.tensor(state)
            action_values = self.dqn(state)
        action = action_values.argmax().item()

        return action

    def remember(self, state: ArrayLike, action: int, reward: int, next_state: ArrayLike, done: bool) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size: int) -> None:
        # Skip if there's not enough memory samples
        if len(self.memory) < batch_size:
            return

        # Unpack the transitions and turn them into tensors
        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: torch.tensor(np.array(x)), zip(*transitions)
        )

        # Calculate target values
        with torch.no_grad():
            targets = self.dqn(states)
            targets[torch.arange(batch_size), actions] = torch.where(
                dones, rewards, rewards + self.gamma * self.dqn(next_states).max(dim=1).values
            )
        
        # Update the DQN
        self.dqn.update(states, targets)

        # Apply epsilon decay
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def train(
        self,
        env: Env,
        n_episodes: int,
        *,
        batch_size: int = 128,
        evaluation_episodes: int = 100,
        apply_best_model: bool = False,
        show_progress: bool = True
    ) -> Any:
        if show_progress:
            progress =tqdm(
                range(n_episodes),
                desc="Training",
                colour="green",
                bar_format="{l_bar}{bar:50}{r_bar}")
        else:
            progress = range(n_episodes)

        best_score = float("-inf")
        best_model = deepcopy(self.dqn.state_dict())

        for _ in progress:
            done = False
            state = env.reset()
            # Play an episode and remember the experience
            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
            
            # Learn from the memory
            self.replay(batch_size)
            
            # Wait for the memory to fill before start evaluating
            if len(self.memory) >= batch_size and evaluation_episodes > 0:
                score = self.evaluate(env, evaluation_episodes)
                if score > best_score:
                    best_score = score
                    best_model = deepcopy(self.dqn.state_dict())

                # Show the score on the progress bar (if enabled)
                if show_progress:
                    progress.set_postfix(score=best_score)
        
        # Load the best model into DQN (if enabled)
        if apply_best_model:
            self.dqn.load_state_dict(best_model)

    def evaluate(self, env: Env, n_episodes: int = 100) -> float:
        eps = self.eps
        self.eps = 0

        score = 0
        for _ in range(n_episodes):
            done = False
            state = env.reset()
            while not done:
                action = self.act(state)
                state, reward, done = env.step(action)
                score += reward
                
        self.eps = eps

        return score / n_episodes

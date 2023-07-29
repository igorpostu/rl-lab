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
    def __init__(self, model: nn.Sequential, criterion: Any, optimiser: Any, *, lr=0.001) -> None:
        super(DQN, self).__init__()

        self.model = model
        self.criterion = criterion()
        self.optimiser = optimiser(self.parameters(), lr=lr)

        self.in_features = model[0].in_features
        self.out_features = model[-1].out_features

    def update(self, x: Tensor, y: Tensor) -> None:
        self.train()
        self.optimiser.zero_grad()
        loss = self.criterion(self(x), y)
        loss.backward()
        self.optimiser.step()
        self.eval()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class DuelingDQN(nn.Module):
    def __init__(
            self,
            F: nn.Sequential,
            V: nn.Sequential,
            A: nn.Sequential,
            criterion: Any,
            optimiser: Any,
            *,
            lr: float = 0.001
    ) -> None:
        super(DuelingDQN, self).__init__()

        self.F = F
        self.V = V
        self.A = A
        
        self.criterion = criterion()
        self.optimiser = optimiser(self.parameters(), lr=lr)
        
        self.in_features = F[0].in_features
        self.out_features = A[-1].out_features

    def update(self, x: Tensor, y: Tensor) -> None:
        self.train()
        self.optimiser.zero_grad()
        loss = self.criterion(self(x), y)
        loss.backward()
        self.optimiser.step()
        self.eval()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def forward(self, x: Tensor) -> Tensor:
        features = self.F(x)
        value = self.V(features)
        advantage = self.A(features)

        return value + advantage - advantage.mean(dim=-1).unsqueeze(dim=-1)

class DQNAgent(Agent):
    """
    A Deep Q-Network (DQN) agent that interacts with an environment, learns from experience, and uses epsilon-greedy policy for action selection.
    ### Parameters:
        - dqn (`DQN`): The DQN model to be used by the agent for action selection and training.
        - memory (`ReplayMemory`): The replay memory buffer used to store and sample experiences for training.
        - gamma (`float`): The discount factor for future rewards in the Q-learning update.
        - eps (`float`): The initial exploration rate for epsilon-greedy action selection.
        - eps_decay (`float`): The decay rate for reducing the exploration rate over time.
        - eps_min (`float`): The minimum exploration rate allowed.
        - double (`bool`): Whether to use double DQN algorithm.
        - tau (`float`): The soft update parameter controlling the interpolation between the policy and target DQNs. Only matters if double DQN is enabled.

    ### Methods:
        - `act`: Selects an action based on the epsilon-greedy policy using the current Q-values from the DQN model.
        - `remember`: Stores a new experience (state, action, reward, next state, and done) in the replay memory.
        - `replay`: Samples experiences from the replay memory and performs a Q-learning update on the DQN model.
        - `train`: Trains the DQN agent in the given environment for the specified number of episodes.
        - `evaluate`: Evaluates the DQN agent in the given environment for the specified number of episodes and returns the average score.
    """
    
    def __init__(
            self,
            dqn: DQN,
            memory: ReplayMemory = ReplayMemory(2000),
            *,
            gamma: float = 0.95,
            eps: float = 1,
            eps_decay: float = 0.995,
            eps_min: float = 0.1,
            double: bool = False,
            tau: float = 0.05
    ) -> None:
        
        self.double = double

        self.policy_dqn = dqn
        self.target_dqn = deepcopy(dqn) if double else dqn
        self.memory = memory

        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.tau = tau

    def act(self, state: ArrayLike) -> int:
        """
        Select an action to take based on a given state.
        ### Parameters:
        - state (`ArrayLike`): The current state of the environment represented as an array-like object.
        ### Returns:
        - The chosen action represented as an integer (`int`)
        """

        # Apply epsilon (random action chance)
        if random.random() <= self.eps:
            return random.randrange(self.policy_dqn.out_features)

        # Choose action
        with torch.no_grad():
            state = torch.tensor(state)
            action_values = self.policy_dqn(state)
            action = action_values.argmax().item()

        return action

    def remember(self, state: ArrayLike, action: int, reward: int, next_state: ArrayLike, done: bool) -> None:
        """
        Store the experience in the agent's replay memory buffer.
        ### Parameters:
        - state (`ArrayLike`): The current state of the environment represented as an array-like object.
        - action (`int`): The action taken by the agent represented as an integer.
        - reward (`int`): The reward received after taking the action.
        - next_state (`ArrayLike`): The resulting state after taking the action represented as an array-like object.
        - done (`bool`): A boolean indicating whether the episode has ended after taking the action.
        ### Returns:
        - `None`
        """

        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size: int) -> None:
        """
        Update the DQN model using a batch of randomly sampled experiences from the replay memory.
        ### Parameters:
        - batch_size (`int`): The number of transitions (experiences) to sample from the replay memory for the update.
        ### Returns:
        - `None`
        """

        # Skip if there's not enough memory samples
        if len(self.memory) < batch_size:
            return

        # Unpack the transitions and turn them into tensors
        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: torch.tensor(np.array(x)), zip(*transitions)
        )

        # Calculate the target values
        with torch.no_grad():
            next_actions = self.policy_dqn(next_states).argmax(dim=-1)
            targets = self.target_dqn(states)
            targets[torch.arange(batch_size), actions] = torch.where(
                dones,
                rewards,
                rewards + self.gamma * self.target_dqn(next_states)[torch.arange(batch_size), next_actions]
            )
        
        # Update the DQNs
        self.policy_dqn.update(states, targets)
        if self.double:
            self.update_target_dqn()

        # Apply epsilon decay
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def train(
        self,
        env: Env,
        episodes: int,
        *,
        batch_size: int = 128,
        evaluation_episodes: int = 100,
        apply_best_model: bool = False,
        show_progress: bool = True
    ) -> List[float]:
        """
        Train the agent using the given environment for a specified number of episodes.
        ### Parameters:
        - env (`Env`): The environment in which the agent will be trained.
        - n_episodes (`int`): The total number of episodes to run during training.
        - batch_size (`int`): The size of the mini-batches used during experience replay.
        - evaluation_episodes (`int`): The number of episodes used for evaluation during training to monitor the agent's performance.
        - apply_best_model (`bool`): Whether to apply the best model found during training to the DQN.
        - show_progress (`bool`): Whether to display the progress bar during training.
        ### Returns:
        - Learning history (`List[float]`)
        """

        if show_progress:
            progress =tqdm(
                range(episodes),
                desc="Training",
                colour="green",
                bar_format="{l_bar}{bar:50}{r_bar}")
        else:
            progress = range(episodes)

        history = []
        best_score = float("-inf")
        best_model = deepcopy(self.policy_dqn.state_dict())

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
            
            # Wait for the memory to fill up before start evaluating
            if evaluation_episodes > 0 and len(self.memory) >= batch_size:
                score = self.evaluate(env, evaluation_episodes)
                history.append(score)
                if score > best_score:
                    best_score = score
                    best_model = deepcopy(self.policy_dqn.state_dict())

                # Show the score on the progress bar (if enabled)
                if show_progress:
                    score_track = best_score if apply_best_model else score
                    progress.set_postfix(score=score_track)

        
        # Load the best model into DQN (if enabled)
        if apply_best_model:
            self.policy_dqn.load_state_dict(best_model)
        
        return history

    def evaluate(self, env: Env, episodes: int = 100) -> float:
        """
        Evaluate the performance of the agent on the given environment by running a specified number of episodes.
        Evaluation runs without exploration (epsilon is set to 0).
        ### Parameters:
        - env (`Env`): The environment in which the agent will be evaluated.
        - n_episodes (`int`): The number of episodes to run for evaluation.
        ### Returns:
        - The evaluation score (`float`)
        """

        eps = self.eps
        self.eps = 0

        score = 0
        for _ in range(episodes):
            done = False
            state = env.reset()
            while not done:
                action = self.act(state)
                state, reward, done = env.step(action)
                score += reward
                
        self.eps = eps

        return score / episodes

    def update_target_dqn(self) -> None:
        for policy_param, target_param in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
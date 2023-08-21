import random
import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from typing import Any, List
from numpy.typing import ArrayLike
from tqdm import tqdm

from components.abstractions import Agent, Env, Memory


class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) model used for Q-value approximation in reinforcement learning.

    ### Parameters:
    - model (`nn.Sequential`): The neural network model representing the DQN architecture.
    - criterion (`Any`): The loss function used for training the DQN.
    - optimiser (`Any`): The optimization algorithm used for updating the DQN's parameters.

    ### Methods:
    - update -> `None`: Perform a single update step using the provided input data and target labels.
    - save -> `None`: Save the DQN's model parameters to a file at the specified path.
    - load -> `None`: Load the DQN's model parameters from a file at the specified path.
    """

    def __init__(self, model: nn.Sequential, criterion: Any, optimiser: Any) -> None:
        super(DQN, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser

        self.in_features = model[0].in_features
        self.out_features = model[-1].out_features

    def update(self, x: Tensor, y: Tensor) -> None:
        """
        Perform a single update step using the provided input data and target labels.
        The method calculates the loss, performs backpropagation, and updates the model's parameters.

        ### Parameters:
        - x (`Tensor`): The input data used for training.
        - y (`Tensor`): The target labels used for training.

        ### Returns:
        - `None`
        """

        self.train()
        self.optimiser.zero_grad()
        loss = self.criterion(self(x), y)
        loss.backward()
        self.optimiser.step()
        self.eval()

    def save(self, path: str) -> None:
        """
        Save the state of the DQN model to a file in the specified path.

        ### Parameters:
        - path (`str`): The file path where the model's state will be saved.

        ### Returns:
        - `None`
        """

        th.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the state of the DQN model from a file in the specified path.
        
        ### Parameters:
        - path (`str`): The file path from where the model's state will be loaded.

        ### Returns:
        - `None`
        """

        self.load_state_dict(th.load(path))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class DuelingDQN(DQN):
    """
    A Dueling Deep Q-Network (Dueling DQN) implementation for reinforcement learning tasks.
    The Dueling DQN architecture separates the value and advantage streams for improved learning.

    ### Parameters:
    - encoder (`nn.Sequential`): The shared feature layers of the model.
    - V (`nn.Sequential`): The value stream layers of the model.
    - A (`nn.Sequential`): The advantage stream layers of the model.
    - criterion (`Any`): The loss function used to calculate the training loss.
    - optimiser (`Any`): The optimizer used to update the model's parameters during training.
    
    ### Methods:
    - save -> `None`: Save the DQN's model parameters to a file at the specified path.
    - load -> `None`: Load the DQN's model parameters from a file at the specified path.
    """

    def __init__(
            self,
            encoder: nn.Sequential,
            V: nn.Sequential,
            A: nn.Sequential,
            criterion: Any,
            optimiser: Any
        ) -> None:
        super().__init__((encoder[0], A[-1]), criterion, optimiser)
        del self.model

        self.encoder = encoder
        self.V = V
        self.A = A

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        value = self.V(features)
        advantage = self.A(features)

        return value + advantage - advantage.mean(dim=-1).unsqueeze(dim=-1)

class DQNAgent(Agent):
    """
    A Deep Q-Network (DQN) agent that interacts with an environment, learns from experience, and uses epsilon-greedy policy for action selection.

    ### Parameters:
    - dqn (`DQN`): The DQN model to be used by the agent for action selection and training.
    - memory (`Memory`): The replay memory buffer used to store and sample experiences for training.
    - gamma (`float`): The discount factor for future rewards in the Q-learning update.
    - eps (`float`): The initial exploration rate for epsilon-greedy action selection.
    - eps_decay (`float`): The decay rate for reducing the exploration rate over time.
    - eps_min (`float`): The minimum exploration rate allowed.

    ### Methods:
    - act -> `int`: Select an action based on the epsilon-greedy policy using the current Q-values from the DQN model.
    - remember -> `None`: Store a new experience (state, action, reward, next state, and done) in the replay memory.
    - replay -> `None`: Sample experiences from the replay memory and performs a Q-learning update on the DQN model.
    - train -> `List[float]`: Train the DQN agent in the given environment for the specified number of episodes.
    - evaluate -> `float`: Evaluate the DQN agent in the given environment for the specified number of episodes and returns the average score.
    """

    def __init__(
            self,
            dqn: DQN,
            memory: Memory,
            *,
            gamma: float = 0.95,
            eps: float = 1,
            eps_decay: float = 0.995,
            eps_min: float = 0.1
    ) -> None:

        self.dqn = dqn
        self.memory = memory

        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def act(self, state: ArrayLike) -> int:
        """
        Select an action to take based on a given state.

        ### Parameters:
        - state (`ArrayLike`): The current state of the environment represented as an array-like object.

        ### Returns:
        - `int`: The chosen action represented as an integer
        """

        # Apply epsilon (random action chance)
        if random.random() <= self.eps:
            return random.randrange(self.dqn.out_features)

        # Choose action
        with th.no_grad():
            state = th.tensor(state)
            action_values = self.dqn(state)

        return action_values.argmax().item()

    def remember(self, *transition) -> None:
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
        state, action, reward, next_state, done = transition
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
            lambda x: th.tensor(np.array(x)), zip(*transitions)
        )

        # Calculate the target values
        targets = self._calculate_targets(batch_size, states, actions, rewards, next_states, dones)
        
        # Update the DQN
        self._update_dqn(states, targets)

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
        - `List[float]`: Learning history
        """

        if show_progress:
            progress = tqdm(
                range(episodes),
                desc="Training",
                colour="green",
                bar_format="{l_bar}{bar:50}{r_bar}")
        else:
            progress = range(episodes)

        history = []
        best_score = float("-inf")
        best_model = deepcopy(self.dqn.state_dict())

        for episode in progress:
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
            
            # Wait for the memory to fill up before starting evaluation
            if evaluation_episodes > 0 and len(self.memory) >= batch_size:
                score = self.evaluate(env, evaluation_episodes)
                history.append(score)
                if score > best_score:
                    best_score = score
                    best_model = deepcopy(self.dqn.state_dict())

                # Show the score on the progress bar (if enabled)
                if show_progress:
                    score_track = best_score if apply_best_model else score
                    progress.set_postfix(score=score_track)

        
        # Load the best model into the DQN (if enabled)
        if apply_best_model:
            self.dqn.load_state_dict(best_model)
        
        return history

    def evaluate(self, env: Env, episodes: int = 100) -> float:
        """
        Evaluate the performance of the agent on the given environment by running a specified number of episodes.
        Evaluation runs without exploration (epsilon is set to 0).
        
        ### Parameters:
        - env (`Env`): The environment in which the agent will be evaluated.
        - n_episodes (`int`): The number of episodes to run for evaluation.
        
        ### Returns:
        - `float`: The evaluation score
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

    def _calculate_targets(self, batch_size: int, *transitions) -> Tensor:
        states, actions, rewards, next_states, dones = transitions

        with th.no_grad():
            targets = self.dqn(states)
            targets[th.arange(batch_size), actions] = th.where(
                dones,
                rewards,
                rewards + self.gamma * self.dqn(next_states).argmax(dim=-1)
            )
        
        return targets

    def _update_dqn(self, states: Tensor, targets: Tensor) -> None:
        self.dqn.update(states, targets)

class DoubleDQNAgent(DQNAgent):
    """
    A Double Deep Q-Network (DQN) agent that interacts with an environment, learns from experience, and uses epsilon-greedy policy for action selection.

    ### Parameters:
    - dqn (`DQN`): The DQN model to be used by the agent for action selection and training.
    - memory (`Memory`): The replay memory buffer used to store and sample experiences for training.
    - gamma (`float`): The discount factor for future rewards in the Q-learning update.
    - eps (`float`): The initial exploration rate for epsilon-greedy action selection.
    - eps_decay (`float`): The decay rate for reducing the exploration rate over time.
    - eps_min (`float`): The minimum exploration rate allowed.
    - tau (`float`): The soft update parameter controlling the interpolation between the policy and target DQNs.

    ### Methods:
    - act -> `int`: Select an action based on the epsilon-greedy policy using the current Q-values from the DQN model.
    - remember -> `None`: Store a new experience (state, action, reward, next state, and done) in the replay memory.
    - replay -> `None`: Sample experiences from the replay memory and performs a Q-learning update on the DQN model.
    - train -> `List[float]`: Train the DQN agent in the given environment for the specified number of episodes.
    - evaluate -> `float`: Evaluate the DQN agent in the given environment for the specified number of episodes and returns the average score.
    """

    def __init__(
        self,
        dqn: DQN,
        memory: Memory,
        *,
        gamma: float = 0.95,
        eps: float = 1,
        eps_decay: float = 0.995,
        eps_min: float = 0.1,
        tau: float = 0.05
    ) -> None:
        super().__init__(dqn, memory, gamma=gamma, eps=eps, eps_decay=eps_decay, eps_min=eps_min)

        self.target_dqn = deepcopy(dqn)
        self.tau = tau

    def _calculate_targets(self, batch_size: int, *transitions) -> Tensor:
        states, actions, rewards, next_states, dones = transitions

        with th.no_grad():
            next_actions = self.dqn(next_states).argmax(dim=-1)
            targets = self.target_dqn(states)
            targets[th.arange(batch_size), actions] = th.where(
                dones,
                rewards,
                rewards + self.gamma * self.target_dqn(next_states)[th.arange(batch_size), next_actions]
            )
        
        return targets

    def _update_dqn(self, states: Tensor, targets: Tensor) -> None:
        self.dqn.update(states, targets)
        for policy_param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

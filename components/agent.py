import random
import numpy as np
import torch as th
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
    """
    A replay memory buffer used to store and sample experiences for training.

    ### Parameters:
    - capacity (`int`): The maximum capacity of the replay memory buffer.

    ### Methods:
    - push -> `None`: Adds a new transition to the replay memory buffer.
    - sample -> `List[Transition]`: Samples a batch of random transitions from the replay memory buffer.
    - save -> `None`: Saves the contents of the replay memory buffer to a file at the specified path.
    - load -> `None`: Loads the contents of the replay memory buffer from a file at the specified path.
    
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
        self.buffer = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        """
        Adds a new transition to the replay memory buffer.

        ### Parameters:
        - state (`ArrayLike`): The current state of the environment represented as an array-like object.
        - action (`int`): The action taken by the agent represented as an integer.
        - reward (`int`): The reward received after taking the action.
        - next_state (`ArrayLike`): The resulting state after taking the action represented as an array-like object.
        - done (`bool`): A boolean indicating whether the episode has ended after taking the action.

        ### Returns:
        - `None`
        """

        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int = 1) -> List[Transition]:
        """
        Samples a batch of random transitions from the replay memory buffer.

        ### Parameters:
        - batch_size (`int`): The number of transitions (experiences) to sample from the replay memory buffer.

        ### Returns:
        - `List[Transition]`: A list of sampled transitions from the replay memory buffer.
        """
        
        return random.sample(self.buffer, batch_size)
    
    def save(self, path: str) -> None:
        """
        Saves the contents of the replay memory buffer to a file at the specified path.

        ### Parameters:
        - path (`str`): The file path where the contents of the replay memory buffer will be saved.

        ### Returns:
        - `None`
        """

        th.save(self.buffer, path)
    
    def load(self, path: str) -> None:
        """
        Loads the contents of the replay memory buffer from a file at the specified path.

        ### Parameters:
        - path (`str`): The file path from where the contents of the replay memory buffer will be loaded.
            
        ### Returns:
        - `None`
        """

        self.buffer.extend(th.load(path))

    def __len__(self) -> int:
        return len(self.buffer)

class DQN(nn.Module):
    """
    A Deep Q-Network (DQN) model used for Q-value approximation in reinforcement learning.

    ### Parameters:
    - model (`nn.Sequential`): The neural network model representing the DQN architecture.
    - criterion (`Any`): The loss function used for training the DQN.
    - optimiser (`Any`): The optimization algorithm used for updating the DQN's parameters.

    ### Methods:
    - save -> `None`: Saves the DQN's model parameters to a file at the specified path.
    - load -> `None`: Loads the DQN's model parameters from a file at the specified path.
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
    - F (`nn.Sequential`): The shared feature layers of the model.
    - V (`nn.Sequential`): The value stream layers of the model.
    - A (`nn.Sequential`): The advantage stream layers of the model.
    - criterion (`Any`): The loss function used to calculate the training loss.
    - optimiser (`Any`): The optimizer used to update the model's parameters during training.
    
    ### Methods:
    - save -> `None`: Saves the DQN's model parameters to a file at the specified path.
    - load -> `None`: Loads the DQN's model parameters from a file at the specified path.
    """

    def __init__(
            self,
            F: nn.Sequential,
            V: nn.Sequential,
            A: nn.Sequential,
            criterion: Any,
            optimiser: Any
        ) -> None:
        super().__init__((F[0], A[-1]), criterion, optimiser)
        del self.model

        self.F = F
        self.V = V
        self.A = A
    
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
    - tau (`float`): The soft update parameter controlling the interpolation between the policy and target DQNs.

    ### Methods:
    - act -> `int`: Selects an action based on the epsilon-greedy policy using the current Q-values from the DQN model.
    - remember -> `None`: Stores a new experience (state, action, reward, next state, and done) in the replay memory.
    - replay -> `None`: Samples experiences from the replay memory and performs a Q-learning update on the DQN model.
    - train -> `List[float]`: Trains the DQN agent in the given environment for the specified number of episodes.
    - evaluate -> `float`: Evaluates the DQN agent in the given environment for the specified number of episodes and returns the average score.
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
            tau: float = 0.05
    ) -> None:

        self.policy_dqn = dqn
        self.target_dqn = deepcopy(dqn) if tau < 1 else dqn
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
    - `int`: The chosen action represented as an integer
        """

        # Apply epsilon (random action chance)
        if random.random() <= self.eps:
            return random.randrange(self.policy_dqn.out_features)

        # Choose action
        with th.no_grad():
            state = th.tensor(state)
            action_values = self.policy_dqn(state)

        return action_values.argmax().item()

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
            lambda x: th.tensor(np.array(x)), zip(*transitions)
        )

        # Calculate the target values
        with th.no_grad():
            next_actions = self.policy_dqn(next_states).argmax(dim=-1)
            targets = self.target_dqn(states)
            targets[th.arange(batch_size), actions] = th.where(
                dones,
                rewards,
                rewards + self.gamma * self.target_dqn(next_states)[th.arange(batch_size), next_actions]
            )
        
        # Update the DQNs
        self.policy_dqn.update(states, targets)
        if self.tau < 1:
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
        best_model = deepcopy(self.policy_dqn.state_dict())

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

        
        # Load the best model into the DQN (if enabled)
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

    def update_target_dqn(self) -> None:
        for policy_param, target_param in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

class RegLoss(nn.Module):
    """
    Regularized Loss (RegLoss) module used for calculating a regularization loss term.
    The RegLoss combines a mean squared error loss and an L1 regularization term.

    ### Parameters:
    - weight (`float`): The weight of the regularization term in the loss calculation.
    """
    def __init__(self, weight: float = 0.1) -> None:
        super(RegLoss, self).__init__()
        self.weight = weight

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return th.mean(self.weight * x + th.pow((x - y), 2))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from components.agent import *
from components.environments import *


def test_play(agent, env):
    """
    Visualize a given agent acting in a given environment.
    """
    eps = agent.eps
    agent.eps = 0
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            print(f"Agent action: {action}")
            state, reward, done = env.step(action)
            
            env.render()
            print(f"Action: {agent.act(state)}")
            _ = input()
    agent.eps = eps

def plot(array):
    plt.plot(array)
    plt.title("Learning History")
    plt.show()

def construct_dqn() -> DQN:
    model = nn.Sequential(
        nn.Linear(9, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 9)
    )
    criterion = nn.MSELoss
    optimiser = optim.Adam

    return DQN(model, criterion, optimiser)

def construct_dueling_dqn() -> DuelingDQN:
    F = nn.Sequential(nn.Linear(9, 9))
    V = nn.Sequential(nn.Linear(9, 1))
    A = nn.Sequential(
        nn.Linear(9, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 9)
    )
    criterion = nn.MSELoss
    optimiser = optim.Adam

    return DuelingDQN(F, V, A, criterion, optimiser)

# Construct DQN
dqn = construct_dqn()

# Define the memory
memory = ReplayMemory(4000)

# Create the agent
agent = DQNAgent(dqn, memory, double=True)

# Define the environment
env = TicTacToeEnv(opponent=agent)

# Train the agent and record the learning history
history = []
history.extend(
    agent.train(env, 2000, batch_size=1024, apply_best_model=True)
)

# Print the agent's learning history
plot(history)

# See the agent play
test_play(agent, env)


# TODO Move "main.py" to a notebook
# TODO Add documentation

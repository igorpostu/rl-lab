import torch.nn as nn
import torch.optim as optim

from components.agent import DQN, ReplayMemory, DQNAgent
from components.environments import *


def test(agent, env):
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


model = nn.Sequential(
    nn.Linear(9, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 9)
)
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

dqn = DQN(model, criterion, optimiser)
memory = ReplayMemory(4000)

agent = DQNAgent(dqn, memory, gamma=0.95, eps_decay=0.99, eps_min=0.15)
env = TicTacToeEnv(opponent=agent)

agent.train(env, 1000, batch_size=1024, evaluation_episodes=0)
agent.train(env, 100, batch_size=1024, evaluation_episodes=100, apply_best_model=True)

print(agent.evaluate(TicTacToeEnv(), 1000))

test(agent, env)


# TODO Double DQN
# TODO Plotting
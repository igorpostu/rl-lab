import torch as th
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from components.environments import *
from components.dqn import *
from components.memory import *
from components.utils import *

set_random_seed(2)


env = CartPoleEnv()

model = nn.Sequential(
    nn.Linear(env.n_observations, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, env.n_actions),
    nn.ReLU()
)

criterion = nn.MSELoss()
optimiser = optim.Adam(get_parameters(model), lr=0.001)
dqn = DQN(model, optimiser)

memory = ReplayMemory(4000)

agent = DoubleDQNAgent(dqn, memory, gamma=0.1)

history = agent.train(env, 500, evaluation_episodes=10)

plt.plot(history)
plt.show()


# TODO Add PERMemory
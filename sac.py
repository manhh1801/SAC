import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as dtb
from copy import deepcopy
import numpy as np
from torch.optim import Adam
import itertools

""" Metadata """
ENV_NAME = "MountainCarContinuous-v0"
FPS = 240
DISCOUNT_RATE = 0.99
ENTROPY_WEIGHT = 0.2
LEARNING_RATE = 0.001
TARGET_WEIGHT = 0.995
LEARNING_DELAY = 10
UPDATE_PERIOD = 100
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1000

""" Define replay memory """
class ReplayMemory():
  def __init__(self, ObsDim, ActDim, MaxSize):
    self.ObsBuf = np.zeros((MaxSize, ObsDim), dtype=np.float32)
    self.ActBuf = np.zeros((MaxSize, ActDim), dtype=np.float32)
    self.RwdBuf = np.zeros(MaxSize, dtype=np.float32)
    self.NextObsBuf = np.zeros((MaxSize, ObsDim), dtype=np.float32)
    self.DoneBuf = np.zeros(MaxSize, dtype=np.float32)
    self.ptr, self.Size, self.MaxSize = -1, 0, MaxSize
  def add(self, Obs, Act, Rwd, NextObs, Done):
    self.ptr += 1
    if(self.ptr == self.MaxSize): self.ptr = 0
    self.Size += 1 if self.Size < self.MaxSize else 0
    self.ObsBuf[self.ptr], self.ActBuf[self.ptr], self.RwdBuf[self.ptr], self.NextObsBuf[self.ptr], self.DoneBuf[self.ptr] = Obs, Act, Rwd, NextObs, Done
  def sample(self, BatchSize):
    indexes = np.random.randint(0, self.Size, size=BatchSize)
    ObsBatch, ActBatch, RwdBatch, NextObsBatch, DoneBatch = self.ObsBuf[indexes], self.ActBuf[indexes], self.RwdBuf[indexes], self.NextObsBuf[indexes], self.DoneBuf[indexes]
    return torch.from_numpy(ObsBatch), torch.from_numpy(ActBatch), torch.from_numpy(RwdBatch), torch.from_numpy(NextObsBatch), torch.from_numpy(DoneBatch)

""" Define critic and actor network """
class ActorNetwork(nn.Module):
  def __init__(self, ObsDim, ActDim, HiddenSizes, Activation, LowerBound, UpperBound):
    super().__init__()
    layers = []
    layers += [nn.Linear(ObsDim, HiddenSizes[0]), Activation()]
    for index in range(len(HiddenSizes) - 1): layers += [nn.Linear(HiddenSizes[index], HiddenSizes[index + 1]), Activation()]
    self.Net = nn.Sequential(*layers)
    self.Mean = nn.Linear(HiddenSizes[-1], ActDim)
    self.LogStdDev = nn.Linear(HiddenSizes[-1], ActDim)
    self.MiddlePoint, self.Dev = (UpperBound + LowerBound) / 2, (UpperBound - LowerBound) / 2
  def forward(self, Obs):
    out = self.Net(Obs)
    mean, std_dev = self.Mean(out), torch.exp(torch.clamp(self.LogStdDev(out), -5, 5))
    ActDtb = dtb.Normal(mean, std_dev)
    Act = ActDtb.rsample()
    LogProbAct = ActDtb.log_prob(Act).sum(axis=-1)
    Act = torch.tanh(Act) * self.Dev + self.MiddlePoint
    return Act, LogProbAct

class CriticNetwork(nn.Module):
  def __init__(self, ObsDim, ActDim, HiddenSizes, Activation):
    super().__init__()
    layers = []
    layers += [nn.Linear(ObsDim + ActDim, HiddenSizes[0]), Activation()]
    for index in range(len(HiddenSizes) - 1): layers += [nn.Linear(HiddenSizes[index], HiddenSizes[index + 1]), Activation()]
    layers.append(nn.Linear(HiddenSizes[-1], 1))
    self.Net = nn.Sequential(*layers)
  def forward(self, Obs, Act):
    QValue = self.Net(torch.cat([Obs, Act], dim=-1))
    return torch.squeeze(QValue, -1)

""" Define loss functions """
def computeQValueLoss(ObsBatch, ActBatch, RwdBatch, NextObsBatch, DoneBatch):
  QValue1 = Critic1(ObsBatch, ActBatch)
  QValue2 = Critic2(ObsBatch, ActBatch)
  with torch.no_grad():
    NextAct, LogProbNextAct = Actor(NextObsBatch)
    NextQValue1 = Critic1Targ(NextObsBatch, NextAct)
    NextQValue2 = Critic2Targ(NextObsBatch, NextAct)
    NextQValue = torch.min(NextQValue1, NextQValue2)
    Target = RwdBatch + DISCOUNT_RATE * (1 - DoneBatch) * (NextQValue - ENTROPY_WEIGHT * LogProbNextAct)
  QValue1Loss = ((QValue1 - Target) ** 2).mean()
  QValue2Loss = ((QValue2 - Target) ** 2).mean()
  return QValue1Loss + QValue2Loss

def computePolicyLoss(ObsBatch):
  Act, LogProbAct = Actor(ObsBatch)
  QValue1 = Critic1(ObsBatch, Act)
  QValue2 = Critic2(ObsBatch, Act)
  QValue = torch.min(QValue1, QValue2)
  PolicyLoss = (ENTROPY_WEIGHT * LogProbAct - QValue).mean()
  return PolicyLoss

""" Define update process """
def updateMainNetworks():
  ObsBatch, ActBatch, RwdBatch, NextObsBatch, DoneBatch = Memory.sample(200)
  CriticOptimizer.zero_grad()
  ActorOptimizer.zero_grad()
  QValueLoss = computeQValueLoss(ObsBatch, ActBatch, RwdBatch, NextObsBatch, DoneBatch)
  QValueLoss.backward(retain_graph=True)
  CriticOptimizer.step()
  for param in critic_params: param.requires_grad = False
  PolicyLoss = computePolicyLoss(ObsBatch)
  PolicyLoss.backward(retain_graph=True)
  ActorOptimizer.step()
  for param in critic_params: param.requires_grad = True

def updateTargetNetworks():
  with torch.no_grad():
    for param, param_targ in zip(critic_params, critic_targ_params):
      param_targ.data.mul_(TARGET_WEIGHT)
      param_targ.data.add_((1 - TARGET_WEIGHT) * param.data)
    for param, param_targ in zip(actor_params, actor_targ_params):
      param_targ.data.mul_(TARGET_WEIGHT)
      param_targ.data.add_((1 - TARGET_WEIGHT) * param.data)

""" Initialize the environment """
Env = gym.make(ENV_NAME, render_mode="human")
Env.metadata["render_fps"] = FPS
ObsDim = Env.observation_space.shape[0]
ActDim = Env.action_space.shape[0]
ActLowerBound, ActUpperBound = Env.action_space.low[0], Env.action_space.high[0]

""" Initialize components """
Memory = ReplayMemory(ObsDim, ActDim, 10000)
Critic1 = CriticNetwork(ObsDim, ActDim, (256, 128, 64), nn.ReLU)
Critic2 = CriticNetwork(ObsDim, ActDim, (256, 128, 64), nn.ReLU)
Actor = ActorNetwork(ObsDim, ActDim, (256, 128, 64), nn.ReLU, torch.tensor(ActLowerBound), torch.tensor(ActUpperBound))
Critic1Targ = deepcopy(Critic1)
Critic2Targ = deepcopy(Critic2)
ActorTarg = deepcopy(Actor)
critic_params = itertools.chain(Critic1.parameters(), Critic2.parameters())
actor_params = Actor.parameters()
critic_targ_params = itertools.chain(Critic1Targ.parameters(), Critic2Targ.parameters())
actor_targ_params = ActorTarg.parameters()
CriticOptimizer = Adam(critic_params, lr = LEARNING_RATE)
ActorOptimizer = Adam(actor_params, lr = LEARNING_RATE)

""" Learn by acting in the environment """
episode = 0
Obs, Act, Rwd, NextObs, Done = None, None, None, Env.reset()[0], None
for episode in range(MAX_EPISODES):
  print(f"Episode {episode + 1}.")
  for step in range(MAX_STEPS_PER_EPISODE):
    Env.render()
    Obs = NextObs
    if episode < LEARNING_DELAY:
      Act = Env.action_space.sample()
    else:
      with torch.no_grad():
        Act, _ = Actor(torch.from_numpy(Obs))
        Act = Act.numpy()
    NextObs, Rwd, terminated, truncated, _ = Env.step(Act)
    Done = terminated or truncated
    Memory.add(Obs, Act, Rwd, NextObs, Done)
    if step % UPDATE_PERIOD == 0 and episode > LEARNING_DELAY:
      updateMainNetworks()
      updateTargetNetworks()
    if(np.float_(Rwd) > 0):
      print("Done.")
      exit(0)
    if Done:
      NextObs = Env.reset()[0]
      continue

""" Close the environment """
Env.close()
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd

import numpy as np

from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import gym
import random
import collections

from google.colab import drive

from policy_module import *
from value_module import *
from custom_env import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Ep_Step=collections.namedtuple('Ep_Step', field_names=['obs','categ','action','reward','obs_f','termin_signal'])

class Agent():
  def __init__(self, env, test_env, gamma, lambd):
    self.env=env
    self.test_env=test_env
    self.gamma=gamma
    self.lambd=lambd

    self.o_dim=self.env.observation_space.shape[0]
    self.a_dim=self.env.action_space.n #discrete action space

    self.pm=Policy_Module(self.o_dim, self.a_dim, [400,300]) #returns categorical object
    self.vm=Value_Module(self.o_dim, [400,300])
  
  def episode_generator(self):
    ep_data=[]
    obs=self.env.reset()
    while True:
      categ=self.pm(obs)
      action=categ.sample()
      obs_f, reward, termin_signal, _=self.env.step(action)
      if termin_signal and obs_f[0]==self.env.goal[0] and obs_f[1]==self.env.goal[1]:
        rts=1 #real termination indicator: not just reaching horizon
      else:
        rts=0
      ep_step=Ep_Step(obs, categ, action, reward, obs_f, rts)
      ep_data.append(ep_step)
      if termin_signal:
        break
    return ep_data
  
  def collect_batch_data(self, batch_size):
    batch_data=[]
    for _ in range(batch_size):
      ep_data=self.episode_generator()
      batch_data.append(ep_data)
    return batch_data
  
class VPG(nn.Module): #Process of General Policy Iteration(GPI)
  def __init__(self, agent):
    super(VPG, self).__init__()
    self.agent=agent
  
  def check_performance(self):
    #w.r.t test env.: run 10 episodes
    len_eps=[]
    acc_rews=[]
    ep_datas=[]
    for _ in range(10):
      obs=self.agent.test_env.reset()
      len_ep=1
      acc_rew=0
      ep_data=[]
      while True:
        action=self.agent.pm(obs)
        action=torch.clamp(action, self.agent.a_low, self.agent.a_high).detach().cpu().numpy()
        obs_f, reward, termin_signal, _=self.agent.test_env.step(action)
        ep_step=Ep_Step(obs, action, reward, obs_f, termin_signal)
        ep_data.append(ep_step)
        acc_rew+=reward
        len_ep+=1
        obs=obs_f
        if termin_signal:
          break
      len_eps.append(len_ep)
      acc_rews.append(acc_rew)
      ep_datas.append(ep_data)
    avg_acc_rew=sum(acc_rews)/10
    avg_len_ep=sum(len_eps)/10
    return avg_acc_rew, avg_len_ep, ep_datas
  
  def get_policy_loss(self, batch_data):
    batch_size=len(batch_data)
    policy_loss=torch.FloatTensor([0]).to(device)
    for ep_data in batch_data:
      GAE=0 #General Advantage Estimator: doesn't require gradient
      ep_policy_loss=torch.FloatTensor([0]).to(device)
      for ep_step in ep_data:
        obs=ep_step.obs
        obs_f=ep_step.obs_f
        action=ep_step.action #scalar value of action index
        reward=ep_step.reward
        categ=ep_step.categ
        rts=ep_step.termin_signal

        log_prob=categ.log_prob(action)
        V=self.agent.vm(obs).detach()
        V_f=self.agent.vm(obs_f).detach()
        if rts:
          tde=reward-V
        else:
          tde=reward+self.agent.gamma*V_f-V
        GAE=GAE*(self.agent.gamma*self.agent.lambd)+tde
        ep_policy_loss=ep_policy_loss-(log_prob*GAE) #negative b.c subject to gradient ascent
      policy_loss=policy_loss+ep_policy_loss
    policy_loss=policy_loss/batch_size
    return policy_loss
  
  def get_value_loss(self, batch_data):
    batch_size=len(batch_data)
    value_loss=torch.FloatTensor([0]).to(device)
    for ep_data in batch_data:
      ep_value_loss=torch.FloatTensor([0]).to(device)
      rtg=0 #reward-to-go: target of updates
      len_ep=len(ep_data)
      for ep_step in ep_data:
        obs=ep_step.obs
        V=self.agent.vm(obs) #requires gradient
        reward=ep_step.reward

        rtg=rtg+reward #finite-horizon undiscounted
        ep_value_loss=ep_value_loss+(V-rtg)**2
      ep_value_loss=ep_value_loss/len_ep
      value_loss=value_loss+ep_value_loss
    value_loss=value_loss/batch_size
    return value_loss
  
  def train(self, batch_size, n_epochs, n_v_epochs, p_lr, v_lr):
    #set optimizers
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)
    value_optim=optim.SGD(self.agent.vm.parameters(), lr=v_lr)

    for epoch in range(1, n_epochs+1):
      #collect trajectories
      batch_data=self.agent.collect_batch_data(batch_size)
      #step1: policy optimization(improvement)
      policy_loss=self.get_policy_loss(batch_data)

      policy_optim.zero_grad()
      policy_loss.backward()
      policy_optim.step() #taking constant steps in policy improvement
      print("Policy_Epoch: {:d}, Policy_loss: {:.3f}".format(epoch, policy_loss.item()))
      #step2: value regression(policy evaluation)
      for v_epoch in range(1, n_v_epochs+1):
        value_loss=self.get_value_loss(batch_data)

        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        if v_epoch%(n_v_epochs/10)==0:
          print("V_Epoch: {:d}, Value_Loss: {:.3f}".format(v_epoch, value_loss.item()))
      #check performance every epoch
      avg_return, avg_len_ep,_=self.check_performance()
      print("Epoch: {:d}, Avg_return: {:.3f}, Avg_len_ep: {:.3f}".format(epoch, avg_return, avg_len_ep))
    return

env=Obs_Wrapper(Windy_Gridworld(grid_size=[7,10], start=[3,0], goal=[3,7], winds=[0,0,0,1,1,1,2,2,1,0]))
test_env=Obs_Wrapper(Windy_Gridworld(grid_size=[7,10], start=[3,0], goal=[3,7], winds=[0,0,0,1,1,1,2,2,1,0]))
agent=Agent(env=env, test_env=test_env, gamma=0.99, lambd=0.97)
vpg=VPG(agent)
vpg.train(batch_size=64, n_epochs=100, n_v_epochs=80, p_lr=3e-4, v_lr=3e-4)
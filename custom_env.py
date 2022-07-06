import numpy as np
import torch
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Windy_Gridworld(gym.Env): #custom environment following openAI gym format
  def __init__(self, grid_size, start, goal, winds): #grid size in row, col order, winds across columns
    super(Windy_Gridworld, self).__init__()
    self.grid_size=grid_size
    self.observation_space=gym.spaces.Box(low=np.array([0,0]), high=np.array([grid_size[0], grid_size[1]]), dtype=np.float32)
    self.action_space=gym.spaces.Discrete(4)

    self.start=start
    self.goal=goal
    self.winds=winds

    self._agent_location=None
    self._step=1 #internal step
    self._horizon=1000 #sets horizon to prevent redundancy

    self.actions=np.array([[0,-1],[0,1],[-1,0],[1,0]]) #4-directional
  
  def _get_obs(self):
    return self._agent_location
  
  def _clip_location(self):
    row=self._agent_location[0]
    col=self._agent_location[1]
    if row<=0:
      self._agent_location[0]=0
    if row>=self.grid_size[0]:
      self._agent_location[0]=self.grid_size[0]-1
    if col<=0:
      self._agent_location[1]=0
    if col>=self.grid_size[1]:
      self._agent_location[1]=self.grid_size[1]-1
    return
  
  def step(self, a_idx):
    action=self.actions[a_idx]
    row=self._agent_location[0]
    col=self._agent_location[1]
    row_f=row+self.winds[col]+action[0]
    col_f=col+action[1]
    if (row_f==self.goal[0] and col_f==self.goal[1]) or self._step>=self._horizon:
      termin_signal=True
    else:
      termin_signal=False
    self._agent_location=np.array([row_f, col_f])
    self._clip_location()
    obs=self._get_obs()
    reward=-1
    info=None
    self._step+=1
    return obs, reward, termin_signal, info

  def reset(self):
    self._agent_location=self.start
    obs=self._get_obs()
    self._step=1
    return obs

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, observation):
    obs=torch.FloatTensor(observation).to(device)
    return obs
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy_Module(nn.Module): #computes logits -> pass through categorical
  def __init__(self, o_dim, a_dim, hidden_sizes):
    super(Policy_Module, self).__init__()
    #create N layers of mlp for output
    layers=[]
    relu=nn.ReLU()
    #first layer
    first=nn.Linear(o_dim, hidden_sizes[0]).to(device)
    layers.append(first)
    layers.append(relu)
    len_h=len(hidden_sizes)
    for h_idx in range(len_h-1):
      linear=nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx+1]).to(device)
      layers.append(linear)
      layers.append(relu)
    #last layer
    last=nn.Linear(hidden_sizes[-1], a_dim).to(device)
    layers.append(last)
    layers.append(relu)
    self.linear_layers=nn.Sequential(*list(layers))
    self.w_sizes, self.b_sizes=self.get_parameter_sizes()
  
  def get_parameter_sizes(self): #initialize linear layers in this part
    w_sizes=[]
    b_sizes=[]
    for element in self.linear_layers:
      if isinstance(element, nn.Linear):
        nn.init.uniform_(element.weight, -(3e-3), (3e-3))
        nn.init.zeros_(element.bias)
        w_s=element.weight.size()
        b_s=element.bias.size()
        w_sizes.append(w_s)
        b_sizes.append(b_s)
    return w_sizes, b_sizes
  
  def vectorize_parameters(self):
    parameter_vector=torch.Tensor([]).to(device)
    for param in self.model.parameters():
      p=param.reshape(-1,1)
      parameter_vector=torch.cat((parameter_vector, p), dim=0)
    return parameter_vector

  def forward(self, observation):
    logits=self.linear_layers(observation)
    cat=Categorical(logits=logits)
    return cat
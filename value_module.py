import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Value_Module(nn.Module):
  def __init__(self, i_size, hidden_sizes):
    super(Value_Module, self).__init__()
    #create N layers of mlp for value output
    layers=[]
    relu=nn.ReLU()
    first=nn.Linear(i_size, hidden_sizes[0]).to(device)
    layers.append(first)
    layers.append(relu)
    len_h=len(hidden_sizes)
    for h_idx in range(len_h-1):
      linear=nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx+1]).to(device)
      layers.append(linear)
      layers.append(relu)
    last=nn.Linear(hidden_sizes[-1], 1).to(device)
    layers.append(last) #no activation function in last layer
    self.linear_layers=nn.Sequential(*list(layers))
    self.element_init()
  
  def element_init(self):
    for element in self.linear_layers:
      if isinstance(element, nn.Linear):
        nn.init.uniform_(element.weight, -(3e-3), 3e-3)
        nn.init.zeros_(element.bias)
    return

  def forward(self, observation):
    value=self.linear_layers(observation)
    return value
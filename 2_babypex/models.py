import math
import random

import ray
import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.arch        = config["fcnet_hiddens"]
        self.input_shape = config["observation_shape"]
        self.num_actions   = config["num_actions"]
        self.activation  = config["fcnet_activation"]

        self.net = nn.Sequential()
        sizes = [self.input_shape] + self.arch + [self.num_actions] + [None]
        for i, (size, size_) in enumerate(zip(sizes, sizes[1:])):
            self.net.add_module(nn.Linear(size, size_))
            if not i == len(sizes) - 1:     #   not last layer
                if self.activation == "relu":
                    self.net.add_module(nn.ReLU())
        
    def forward(self, x):
        return self.net(x)

def get_q_network(config):
    q_net = QNetwork(config)
    return q_net

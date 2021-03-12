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
        sizes = [self.input_shape[0]] + self.arch + [self.num_actions]
        for i, (size, size_) in enumerate(zip(sizes, sizes[1:])):
            self.net.add_module(name="fc"+str(i), module=nn.Linear(size, size_))
            if i < len(sizes) - 2:  #   last layer should have no activation function
                if self.activation == "relu":
                    self.net.add_module(name="af"+str(i), module=nn.ReLU())
        
    def forward(self, x):
        return self.net(x)

def get_q_network(config):
    q_net = QNetwork(config)
    return q_net

if __name__ == "__main__":
    config = {
        "env": "CartPole-v0",
        "num_actions": 2,
        "observation_shape": (4,),

        "target_network_update_interval": 500,
        "num_workers": 2,
        "eval_num_workers": 2,
        "eval_device": "cpu",
        "train_device": "gpu",
        "n_step": 3,
        "max_eps": 0.5,
        "train_batch_size": 512,
        "gamma": 0.99,
        "fcnet_hiddens": [256, 256, 256],
        "fcnet_activation": "relu",
        "lr": 1e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 5_000,
        "timesteps_per_iteration": 10_000,
        "grad_clip": 10,
    }

    net = QNetwork(config)
    print(net.net)
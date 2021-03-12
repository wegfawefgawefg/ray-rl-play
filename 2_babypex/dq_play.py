import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from checkpoints.model_checkpoint_backup_config import config
from models import QNetwork

def get_env_configs(config):
    env = gym.make(config["env"])
    config["num_actions"] = env.action_space.n
    config["observation_shape"] = env.observation_space.shape
    return config

if __name__ == '__main__':
    config = get_env_configs(config)

    env = gym.make('CartPole-v1').unwrapped
    net = QNetwork(config)
    print(net.net)
    net.load_state_dict(torch.load("checkpoints/model_checkpoint_backup"))

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0)
            qs = net(state)[0]
            action = torch.argmax(qs).item()
            state_, reward, done, info = env.step(action)


            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1

import torch
import ray
import numpy as np

from actor import Actor
from replay_buffer import ReplayBuffer
from parameter_server import ParameterServer
from models import models
from learner import Learner

max_samples = 500000
config = {
    "env": "CartPole-v0",
    "num_actions": 2,
    "observation_shape": (4,),

    "target_network_update_interval": 500,
    "num_workers": 50,
    "eval_num_workers": 10,
    "eval_device": "cpu",
    "train_device": "gpu",
    "n_step": 3,
    "max_eps": 0.5,
    "train_batch_size": 512,
    "gamma": 0.99,
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
    "lr": 1e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 5_000,
    "timesteps_per_iteration": 10_000,
    "grad_clip": 10,
}


ray.init()

parameter_server = ParameterServer.remote(config)
replay_buffer = ReplayBuffer.remove(config)
learner = Learner.remote(config, replay_buffer, parameter_server)

train_actor_ids = []
eval_actor_ids = []

learner.start_learning.remote()

#   start train actors
for i in range(config["num_workers"]):
    epsilon = config["max_eps"] * i / config["num_workers"]
    training_actor = Actor.remote(
        "train-" + str(i),
        replay_buffer, 
        parameter_server, 
        config, 
        epsilon)
    training_actor.sample.remote()
    train_actor_ids.append(training_actor)

#   start eval actors
for i in range(config["eval_num_workers"]):
    epsilon = 0
    eval_actor = Actor.remote(
        "eval-" + str(i),
        replay_buffer,
        parameter_server,
        config,
        epsilon,
        eval=True)
    eval_actor_ids.append(eval_actor)

#   fetch samples in loop and sync actor weights
total_samples = 0
best_eval_mean_reward = np.NINF
eval_mean_rewards = []
while total_samples < max_samples:
    new_total_samples = ray.get( replay_buffer.get_total_env_samples.remote() )
    num_new_samples = new_total_samples - total_samples
    if num_new_samples >= config["timesteps_pre_iteration"]:
        total_samples = new_total_samples
        
        parameter_server.set_eval_weights.remote()
        eval_sampling_ids = [eval_actor.sample.remote() for eval_actor in eval_actor_ids]
        eval_rewards = ray.get(eval_sampling_ids)
        eval_mean_reward = np.mean(eval_rewards)

        eval_mean_rewards.append(eval_mean_reward)
        if eval_mean_reward > best_eval_mean_reward:
            best_eval_mean_reward = eval_mean_reward
            parameter_server.save_eval_rewards.remote()
        
[actor.stop.remote() for actor in train_actor_ids]
learner.stop.remote()
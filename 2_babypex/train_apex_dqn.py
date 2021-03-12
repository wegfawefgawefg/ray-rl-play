import os

import ray
import gym
import torch
import numpy as np


from actor import Actor
from replay_buffer import ReplayBuffer
from parameter_server import ParameterServer
from learner import Learner

# os.environ['GRPC_VERBOSITY']='DEBUG'
# os.environ['http_proxy']='some proxy'
# os.environ['https_proxy']='some proxy'

def get_env_configs(config):
    env = gym.make(config["env"])
    config["num_actions"] = env.action_space.n
    config["observation_shape"] = env.observation_space.shape
    return config

def main(config, max_samples):
    get_env_configs(config)
    ray.init()

    parameter_server = ParameterServer.remote(config)
    replay_buffer = ReplayBuffer.remote(config)
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
        total_env_samples_id = replay_buffer.get_total_env_samples.remote()
        new_total_samples = ray.get(total_env_samples_id)
        num_new_samples = new_total_samples - total_samples
        if num_new_samples >= config["timesteps_per_iteration"]:
            total_samples = new_total_samples
            print("Total samples:", total_samples)
            parameter_server.set_eval_weights.remote()
            eval_sampling_ids = [eval_actor.sample.remote() for eval_actor in eval_actor_ids]
            eval_rewards = ray.get(eval_sampling_ids)
            print("Evaluation rewards: {}".format(eval_rewards))
            eval_mean_reward = np.mean(eval_rewards)
            eval_mean_rewards.append(eval_mean_reward)
            print("Mean evaluation reward: {}".format(eval_mean_reward))
            if eval_mean_reward > best_eval_mean_reward:
                print("Model has improved! Saving the model!")
                best_eval_mean_reward = eval_mean_reward
                parameter_server.save_eval_weights.remote()
            
    print("Finishing the training.\n\n\n\n\n\n")
    [actor.stop.remote() for actor in train_actor_ids]
    learner.stop.remote()

if __name__ == "__main__":
    max_samples = 5_000_000
    config = {
        "env": "CartPole-v0",
        "target_network_update_interval": 512,
        "num_workers": 50,
        "eval_num_workers": 10,
        "eval_device": "cpu",
        "train_device": "cpu",
        "n_step": 3,
        "max_eps": 0.5,
        "train_batch_size": 32,
        "gamma": 0.75,
        "fcnet_hiddens": [1024, 512],
        "fcnet_activation": "relu",
        "lr": 1e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 5_000,
        "timesteps_per_iteration": 32,
        "grad_clip": 10,

        "q_update_freq": 256,
        "send_experience_freq": 256,
    }
    main(config, max_samples)
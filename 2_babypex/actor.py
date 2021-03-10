from collections import deque

import torch
import gym
import ray
import numpy as np
from ray import actor
from ray import parameter

from models import get_q_network

@ray.remote
class Actor:
    def __init__(self,
            actor_id,
            replay_buffer, 
            parameter_server,
            config, 
            epsilon,
            eval=False):

        self.actor_id         = actor_id
        self.replay_buffer    = replay_buffer
        self.parameter_server = parameter_server
        self.config           = config
        self.epsilon          = self.epsilon
        self.eval             = eval

        self.device = torch.device("cpu")
        if config["eval_device"] == "gpu":
            if torch.cuda_is_available():
                self.device = torch.device("cuda:0")
        
        self.obs_shape            = config["obs_shape"]
        self.num_actions            = config["num_actions"]
        self.multi_step_n         = config.get("n_step", 1)
        self.q_update_freq        = config.get("q_update_freq", 100)
        self.send_experience_freq = config.get("send_experience_freq", 100)
        
        self.q_net = get_q_network(config)
        if self.eval:
            self.q_net.eval()
        else:
            self.q_net.train()
        self.q_net.to(self.device)

        self.env = gym.make(config["env"])
        self.local_buffer = []

        self.continue_sampling = True
        self.cur_episodes = 0
        self.cur_steps = 0

        def update_q_network(self):
            if self.eval:
                state_dict_id = self.parameter_server.get_eval_weights.remote()
            else:
                state_dict_id = self.parameter_server.get_weights.remote()
            new_state_dict = ray.get(state_dict_id)
            if new_state_dict:
                self.q_net.load(new_state_dict)
            else:
                print("Weights are not available yet, skipping.")

        def get_action(self, state):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            state = state.unsqueeze(0)
            qs = self.q_net(state)[0]
            if np.random.uniform() <= self.eps:
                action = np.random.randint(self.num_actions)
            else:
                action = torch.argmax(qs).item()
            return action

        def get_n_step_trans(self, n_step_buffer):
            gamma = self.config["gamma"]
            discounted_return = 0
            cum_gamma = 1

            for state, action, reward, done in list(n_step_buffer)[:-1]:
                discounted_return += cum_gamma * reward
                cum_gamma *= gamma
            state,      action, _,     _ = n_step_buffer[0]
            last_state,      _, _,  done = n_step_buffer[-1]
            experience = (state, action, discounted_return, last_state, done, cum_gamma)

            return experience

        def stop(self):
            self.continue_sampling = False

        def sample(self):
            print("Starting sampling in actor {}".format(self.actor_id))
            self.update_q_network()
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            n_step_buffer = deque(maxlen=self.multi_step_n + 1)

            while self.continue_sampling:
                action = self.get_action(state)
                state_, reward, done, info = self.env.step(action)
                
                partial_transition = (state, action, reward, done)
                n_step_buffer.append(partial_transition)

                if len(n_step_buffer) == self.multi_step_n + 1:
                    self.local_buffer.append(
                        self.get_n_step_trans(n_step_buffer))
                
                self.cur_steps += 1
                episode_reward += reward
                episode_length += 1

                if done:
                    if self.eval:
                        break
                    state_ = self.env.reset()
                    if len(n_step_buffer) > 1:
                        '''if the done came and the episode is less than n steps, 
                            you atleast get 2 or more steps considered in this transition'''
                        self.local_buffer.append(
                            self.get_n_step_trans(n_step_buffer))
                    self.cur_episodes += 1
                    episode_reward = 0
                    episode_length = 0

                state = state_

                if not self.eval:
                    if self.cur_steps % self.send_experience_freq == 0:
                        self.send_experience_to_replay()
                    if self.cur_steps % self.q_update_freq == 0:
                        self.update_q_network()

        def send_experience_to_replay(self):
            result_id = self.replay_buffer.add.remote(self.local_buffer)
            ray.wait([result_id])
            self.local_buffer = []
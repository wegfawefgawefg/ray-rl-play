import copy
import time

import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import get_q_network

@ray.remote
class Learner:
    def __init__(self, config, replay_buffer, parameter_server):
        self.config = config
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server

        self.device = torch.device("cpu")
        if config["eval_device"] == "gpu":
            if torch.cuda_is_available():
                self.device = torch.device("cuda:0")

        self.q_net = get_q_network(config).train().to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.train_batch_size = config["train_batch_size"]
        self.total_collected_samples = 0
        self.samples_since_last_update = 0

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.loss = nn.MSELoss()

        self.send_weights()
        self.stopped = False

    def send_weights(self):
        id = self.parameter_server.update_weights.remote(self.q_net.state_dict())
        ray.get(id)

    def stop(self):
        self.stopped = True

    def start_learning(self):
        print("Learning starting...")
        self.send_weights()
        while not self.stopped:
            total_sample_count_id = self.replay_buffer.get_total_env_samples.remote()
            num_total_samples = ray.get(total_sample_count_id)
            if num_total_samples >= self.config["learning_starts"]:
                self.learn()

    def learn(self):
        samples_id = self.replay_buffer.sample.remote(self.train_batch_size)
        samples = ray.get(samples_id)

        if not samples:
            print("No samples received from the buffer.")
            time.sleep(5)
            return False

        if samples:
            batch_size = len(samples)
            self.total_collected_samples += batch_size
            self.samples_since_last_update += batch_size
            num_actions = self.config["num_actions"]
            state_shape = self.config["observation_shape"]
        
            #   experience = (state, action, discounted_return, last_state, done, cum_gamma)
            states, actions, rewards, states_, dones, gammas = list(zip(*samples))
            states  = torch.tensor( states , dtype=torch.float32).to(self.device)
            actions = torch.tensor( actions, dtype=torch.long   ).to(self.device)
            rewards = torch.tensor( rewards, dtype=torch.float32).to(self.device)
            states_ = torch.tensor( states_, dtype=torch.float32).to(self.device)
            dones   = torch.tensor( dones,   dtype=torch.bool   ).to(self.device)
            gammas  = torch.tensor( gammas,  dtype=torch.float32).to(self.device)

            batch_indices = np.arange(batch_size, dtype=np.int64)
            action_qs = self.q_net(states)[batch_indices, actions]    #   (batch_size, 1)

            policy_qs = self.q_net(states_)             #   (batch_size, num_actions)
            actions_ = torch.max(policy_qs, dim=1)[1]   #   (batch_size, 1)
            
            qs_       = self.target_q_net(states_)      #   (batch_size, num_actions)
            action_qs_ = qs_[batch_indices, actions_]
            action_qs_[dones] = 0.0

            q_targets = rewards + gammas * action_qs_

            
            loss = self.loss(q_targets, action_qs).to(self.device)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.send_weights()

            if self.samples_since_last_update > self.target_network_update_interval:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
                self.samples_since_last_update = 0
            return True

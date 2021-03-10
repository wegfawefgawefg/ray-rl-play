from collections import deque

import ray
import numpy as np

@ray.remote
class ReplayBuffer:
    def __init__(self, config):
        self.replay_buffer_size = config["buffer_size"]
        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.total_env_samples = 0

    def add(self, fresh_transitions):
        for transition in fresh_transitions:
            self.buffer.append(transition)
            self.total_env_samples += 1
        return True

    def sample(self, num_samples):
        if len(self.buffer) > num_samples:
            sample_indices = np.random.randint(len(self.buffer), size=num_samples)
            return [self.buffer[index] for index in sample_indices]

    def get_total_env_samples(self):
        return self.total_env_samples
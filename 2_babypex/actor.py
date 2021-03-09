import ray
import numpy as np
from ray import actor
from ray import parameter

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
        
        self.Q = get_Q_network(config)
        self.env = gym.make(config["env"])
        self.local_buffer = []
        
        self.obs_shape            = config["obs_shape"]
        self.n_actions            = config["n_actions"]
        self.multi_step_n         = config.get("n_step", 1)
        self.q_update_freq        = config.get("q_update_freq", 100)
        self.send_experience_freq = config.get("send_experience_freq", 100)

        self.continue_sampling = True
        self.cur_episodes = 0
        self.cur_steps = 0

        def update_q_network(self):
            if self.eval:
                params_id = self.parameter_server.get_eval_weights.remote()
            else:
                params_id = self.parameter_server.get_weights.remote()
            new_weights = ray.get(params_id)
            if new_weights:
                self.Q.set_weights(new_weights)

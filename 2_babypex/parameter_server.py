import ray
import torch

from models import get_q_network

'''TODO:
replace the weight saving feature
'''

@ray.remote
class ParameterServer:
    def __init__(self, config):
        self.weights = None         # pytorch param state_dict
        self.eval_weights = None    # pytorch param state_dict

    def update_weights(self, new_parameters):
        ''' expects state_dict'''
        self.weights = new_parameters
        return True

    def set_eval_weights(self):
        ''' take snapshot of training weights
                for the eval agents'''
        self.eval_weights = self.weights
        return True

    def get_weights(self):
        ''' training agents get these   '''
        # assert self.weights is not None
        return self.weights

    def get_eval_weights(self):
        ''' eval agents get these   '''
        # assert self.eval_weights is not None
        return self.eval_weights

    def save_eval_weights(self, 
            filename= "checkpoints/model_checkpoint"):
        torch.save(self.eval_weights, filename)
        print("Saved.")

'''
# This environment model provides a API for learning agent to interact with SUMO simulation

To-do
'''
import torch

class SUMOAPI:
    def __init__(self) -> None:
        pass

    def get_next_state(self, args):
        return 

    def compute_reward(self, args):
        return
    
    def perform_action(self, action)->list[torch.Tensor, torch.Tensor]:
        next_state = self.get_next_state()
        reward = self.compute_reward()
        return next_state, reward

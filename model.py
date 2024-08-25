import torch
import torch.nn as nn
from torch.nn import functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=None, *args, **kwargs) -> None:
        super(Actor, self).__init__(*args, **kwargs)

        self.input = nn.Linear(state_dim, state_dim)
        self.fc1 = nn.Linear(state_dim, state_dim)
        self.fc2 = nn.Linear(state_dim, action_dim)
        self.output = nn.Softmax(action_dim, dim=-1)


    def forward(self, state):
        out = self.input(state)
        out = F.relu(out)

        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.output(out)

        return out
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, *args, **kwargs) -> None:
        super(Critic, self).__init__(*args, **kwargs)

        self.state_input = nn.Linear(state_dim,state_dim)
        self.action_input = nn.Linear(action_dim, action_dim)

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_out = self.state_input(state)
        action_out = self.action_input(action)

        out = self.fc1(torch.cat(state_out, action_out),0)

        out = self.output(out)

        return out


class ActorCriticModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, *args, **kwargs) -> None:
        super(ActorCriticModel, self).__init__(*args, **kwargs)

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)

    def forward(self, state):
        action = self.actor(state)

        value_est = self.critic(state, action)

        return action, value_est


import torch
import torch.multiprocessing as mp
from torch.optim import Adam

from model import ActorCriticModel
from environment import SUMOAPI
from utils.optimizer import SharedAdam

MAX_COUNTER = 10000
MAX_STEP = 1000

def train_submodel(rank, num_state, num_action, critic_hidden_size, shared_model:torch.nn.Module, counter, lock, optimizer:SharedAdam, gamma):
    '''
    # Train a sub-model

    To-do
    '''
    torch.manual_seed(1 + rank)

    # Load environment (each submodel interacts with different instance of environment)
    env = SUMOAPI()

    submodel = ActorCriticModel(num_state, num_action, shared_model, critic_hidden_size)
    thread_step_counter = 1

    submodel.train()


    while counter <= MAX_COUNTER:
        optimizer.zero_grad()
        submodel.load_state_dict(shared_model.state_dict())
        t_start = thread_step_counter
        state = env.get_state()

        rewards = []
        values = []
        action_probs = []
        for step in range(t_start,MAX_STEP):
            counter += 1
            action_prob, value_est = submodel(state)

            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()

            next_state, reward = env.perform_action(action)

            action_probs.append(action_prob)
            values.append(value_est)
            rewards.append(reward)
            
            state = next_state

        R = submodel.critic(state)

        actor_loss, critic_loss = 0, 0
        log_probs = torch.log(torch.Tensor(action_probs))

        for i in reversed(range(len(rewards))):
            R = reward[i] + gamma*R 
            
            advantage = reward[i] - values[i]
            actor_loss += -log_probs[i]*(advantage)
            critic_lost += (advantage)**2

        total_loss = actor_loss + critic_loss
        total_loss.backward()
        
        for submodel_param, shared_model_param in zip(submodel.parameters(), shared_model.parameters()):
            shared_model._grad = submodel_param._grad

        optimizer.step()

def train():
    '''
    # Train Asynchronous Advantage Actor-Critic
    '''
    # Define parameters
    num_submodel = 4
    num_state = 4
    num_action = 4
    critic_hidden_size = 64


    # Load model
    ## Create shared model
    shared_model = ActorCriticModel(num_state, num_action, critic_hidden_size)
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=0.9)
    optimizer.share_memory()

    ## Create sub-model
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    processes = []

    for rank in range(num_submodel):
        p = mp.Process(target=train_submodel, args=(rank, shared_model, counter, lock, optimizer,))
        p.start()
        processes.append(p)

    # Train
    for p in processes:
        p.join()

    # Save model
    shared_model._save_to_state_dict('')
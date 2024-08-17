import torch
import torch.multiprocessing as mp

from model import ActorCriticModel
from environment import SUMOAPI
from utils.optimizer import SharedAdam

def train_submodel(rank, num_state, num_action, critic_hidden_size, shared_model:torch.nn.Module, counter, lock, optimizer):
    '''
    # Train a sub-model

    To-do
    '''
    torch.manual_seed(1 + rank)

    # Load environment (each submodel interacts with different instance of environment)
    env = SUMOAPI()

    submodel = ActorCriticModel(num_state, num_action, shared_model, critic_hidden_size)

    submodel.train()

    state = env.get_state()

    while True:
        submodel.load_state_dict(shared_model.state_dict())

        for step in range():
            action, value_est = submodel(state)

            # To-do

            

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
        p = mp.Process(target=train_submodel, args=(rank, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)

    # Train
    for p in processes:
        p.join()

    # Save model
    shared_model._save_to_state_dict('')
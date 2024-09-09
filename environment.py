'''
# This environment model provides a API for learning agent to interact with SUMO simulation

To-do
'''
import torch
import traci
import os
import sys

class SUMOAPI:
    def __init__(self, path_sumo_cfg, max_step=1000) -> None:
        try:
            traci.start(['sumo', '-c', path_sumo_cfg])  # Ignore TL jams
            print('Started SUMO and TraCi successfully!')
        except Exception as e:
            print("An error has occurred when trying to start SUMO and TraCI!")
            print(e)
            traci.close()
            sys.exit(1)

        self.traffic_light_ids = traci.trafficlight.getIDList()
        self.step = 0
        self.max_step = max_step


    def transform_to_next_state(self, *args)->torch.Tensor:
        try:
            traci.simulationStep()
            # To-do: Code get_state() 
            self.state = self.get_state()
            self.step += 1
            print(f"Step {self.step}: Simulation running.")
            return self.state
        
        except Exception as e:
            print(f"Error during simulation step: {e}")
            traci.close()
            sys.exit(1)

    def get_state(self, *args)->torch.Tensor:
        return 

    def compute_reward(self, state, *args)->torch.Tensor:
        return


    def perform_action(self, action) -> list[torch.Tensor, torch.Tensor]:
        '''
        Perform an action and return next state, reward
        '''
        if self.step < self.max_step:
            # To-do: Code transform_to_next_state(), compute_reward()
            next_state = self.transform_to_next_state(action)
            reward = self.compute_reward(action)
            return next_state, reward

        else:
            print("Reached termination!")


if __name__ == "__main__":
    print('Running demo!')
    path_sumo_cfg = 'networks/osm.sumocfg'
    sumo = SUMOAPI(path_sumo_cfg)

    # # Test environment
    # for t in range(1):
    #     for tls in sumo.traffic_light_ids[:1]:
    #         light_phase = traci.trafficlight.getRedYellowGreenState(tls)
    #         controlled_lane = traci.trafficlight.getControlledLanes(tls)
    #         print(f'Lane {controlled_lane}: {light_phase}\n')

        

        

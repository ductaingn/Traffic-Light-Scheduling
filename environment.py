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
        self.state = None  #Initial state


    def transform_to_next_state(self, action)->torch.Tensor:
        try:
            self.perform_action(action)
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
        states = []
        for tls in self.traffic_light_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls)
            for lane in controlled_lanes:
                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                density = traci.lane.getLastStepOccupancy(lane)
                states.append([queue_length, density])

            current_phase = traci.trafficlight.getPhase(tls)
            num_phases = traci.trafficlight.getPhaseNumber(tls)
            phase_one_hot = [1 if i == current_phase else 0 for i in range(num_phases)]
            states.append(phase_one_hot)

        return torch.tensor(states, dtype=torch.float32)

    def compute_reward(self, state, *args)->torch.Tensor:
        queue_lengths = state[:, 0]
        total_queue_length = torch.sum(queue_lengths)
        reward = -total_queue_length**2
        return reward


    def perform_action(self, action) -> list[torch.Tensor, torch.Tensor]:
        '''
        Perform an action and return next state, reward
        '''
        for tls in self.traffic_light_ids:
            if action == 0:
                traci.trafficlight.setPhase(tls, 0) #đèn xanh hướng B-N
            elif action == 1:
                traci.trafficlight.setPhase(tls, 1) #đèn xanh hướng Đ-T
            elif action == 2:
                traci.trafficlight.setPhase(tls, 2) #đèn xanh rẽ phải hướng B-N
            elif action == 3:
                traci.trafficlight.setPhase(tls, 3) #đèn xanh rẽ phải hướng Đ-T

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

        

        

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
            print(f"State before action: {self.state}")
            self.perform_action(action)

            # To-do: Code get_state() 
            self.state  = self.get_state()
            print(f"State after action: {self.state }")
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

            phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
            num_phases = len(phases[0].phases)
            current_phase = traci.trafficlight.getPhase(tls)
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
                traci.trafficlight.setPhase(tls, 0)  # Đèn xanh hướng B-N
                traci.simulationStep()  

                traci.trafficlight.setPhase(tls, 4)  # Pha đèn vàng hướng B-N
                for _ in range(3):  
                    traci.simulationStep()
            elif action == 1:
                traci.trafficlight.setPhase(tls, 1)  # Đèn xanh hướng Đ-T
                traci.simulationStep()  
            
                traci.trafficlight.setPhase(tls, 5) 
                for _ in range(3):  
                    traci.simulationStep()
            elif action == 2:
                traci.trafficlight.setPhase(tls, 2)  # Đèn xanh rẽ trái hướng B-N
                traci.simulationStep()  
            
                traci.trafficlight.setPhase(tls, 6)  
                for _ in range(3):  
                    traci.simulationStep()
            elif action == 3:
                traci.trafficlight.setPhase(tls, 3)  # Đèn xanh rẽ trái hướng Đ-T
                traci.simulationStep() 
        
                traci.trafficlight.setPhase(tls, 7) 
                for _ in range(3): 
                    traci.simulationStep()

if __name__ == "__main__":
    print('Running demo!')
    path_sumo_cfg = 'networks/double.sumocfg'
    sumo = SUMOAPI(path_sumo_cfg)

    # Test environment
    for t in range(100):
        for tls in sumo.traffic_light_ids[:1]:
        #     light_phase = traci.trafficlight.getRedYellowGreenState(tls)
        #     controlled_lane = traci.trafficlight.getControlledLanes(tls)
        #     print(f'Lane {controlled_lane}: {light_phase}\n')
            action = t % 4 
            state = sumo.transform_to_next_state(action)
            reward = sumo.compute_reward(state)
            print(f'Step {t}, Action: {action}, Reward: {reward}')

    traci.close()

        

        

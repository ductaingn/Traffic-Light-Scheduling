import traci
import torch
from SUMO_simulation import SumoSim
from vehicles import Vehicles
from network_data import NetworkData
from traffic_signal_controller import RLTrafficSignalController, UniformFixedTrafficSignalController, FixedTrafficSignalController, Intersection
from reinforvement_learning_agent import ReinforcementLearningAgent

class SUMOAPI:
    def __init__(self, sumo_cfg_file, agent_params, network_data):
        # Initialize the SUMO simulation
        self.sumo_sim = SumoSim(sumo_cfg_file)

        # Initialize the agent
        self.agent = ReinforcementLearningAgent(agent_params)

        # Initialize the network data
        self.network_data = network_data

        # Initialize traffic signal controllers and vehicle manager
        self.traffic_signal_controllers = [
            RLTrafficSignalController("intersection_1", self.network_data, self.sumo_sim, agent_params),
            FixedTrafficSignalController("intersection_2", self.network_data, self.sumo_sim)
        ]
        self.vehicle_manager = Vehicles(self.network_data)

    def start_simulation(self):
        self.sumo_sim.start()

    def stop_simulation(self):
        self.sumo_sim.stop()

    def step_simulation(self):
        self.sumo_sim.step()

    def get_state(self):
        # Example: Get the state of traffic signals and vehicles
        signal_states = [tsc.get_state() for tsc in self.traffic_signal_controllers]
        vehicle_states = self.vehicle_manager.get_vehicle_states()

        return {
            'signals': signal_states,
            'vehicles': vehicle_states
        }

    def take_action(self, actions):
        # Apply actions to traffic signal controllers
        for tsc, action in zip(self.traffic_signal_controllers, actions):
            tsc.apply_action(action)

    def run_episode(self):
        self.start_simulation()
        done = False
        while not done:
            state = self.get_state()
            action = self.agent.select_action(state)
            self.take_action(action)
            self.step_simulation()
            reward, done = self.agent.evaluate(state)
            self.agent.update(state, action, reward)
        self.stop_simulation()

    def reset(self):
        self.sumo_sim.reset()

if __name__ == "__main__":
    sumo_api = SUMOAPI(r"C:\Traffic-Light-Scheduling\networks\osm.sumocfg", agent_params={}, network_data={})
    sumo_api.run_episode()

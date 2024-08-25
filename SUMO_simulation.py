# SUMO simulation management and integration

import sys, os, subprocess, time

import sys, os, subprocess, time

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                                 "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
from vehicles import Vehicles
from network_data import NetworkData
from traffic_signal_controller import RLTrafficSignalController, UniformFixedTrafficSignalController, FixedTrafficSignalController, Intersection
import numpy as np


class SumoSim():
    def __init__(self, port, idx, cfg_fp):
        self.port = port
        self.idx = idx
        self.cfg_fp = cfg_fp

    def input_to_one_hot(self, phases):
        identity = np.identity(len(phases))
        one_hots = {phases[i]: identity[i, :] for i in range(len(phases))}
        return one_hots

    def int_to_input(self, phases):
        return {p: phases[p] for p in range(len(phases))}

    def action_state_lanes(self, actions, index_to_lane):
        action_state_lanes = {a: [] for a in actions}
        for a in actions:
            for s in range(len(a)):
                if a[s] == 'g' or a[s] == 'G':
                    lane = index_to_lane.get(s)
                    if lane:  # Kiểm tra sự tồn tại của làn đường
                        if self.conn.lane.getLength(lane) > 0:  # Kiểm tra nếu làn đường không bị vô hiệu hóa
                            action_state_lanes[a].append(lane)
                        else:
                            print(f"Lane {lane} is disabled, skipping.")
            action_state_lanes[a] = list(set(action_state_lanes[a]))
        return action_state_lanes

    def gen_sim(self, nogui, sim_len):
        if nogui is False:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        ### Khởi động SUMO và kết nối
        self.sumo_process = subprocess.Popen(
            [sumoBinary, "-c", self.cfg_fp, "--remote-port", str(self.port), "--no-warnings", "--no-step-log",
             "--random"], stdout=None, stderr=None)

        ### Đợi cho SUMO hoàn toàn khởi động
        time.sleep(7)

        ### Kết nối đến SUMO qua cổng đã chỉ định
        try:
            self.conn = traci.connect(self.port)
            print(f"Connected to SUMO on port {self.port}")
        except traci.exceptions.FatalTraCIError as e:
            print(f"Failed to connect to SUMO: {e}")
            sys.exit(1)

        self.sim_len = sim_len
        self.t = 0

    def get_tsc_data(self):
        tsc_list = self.conn.trafficlight.getIDList()
        tsc_data = {_id: {} for _id in tsc_list}

        for tsc in tsc_data:
            # print('----- '+tl)
            ###get green phases for TL
            tsc_logic = self.conn.trafficlight.getAllProgramLogics(tsc)[0]
            ##http://www.sumo.dlr.de/pydoc/traci._trafficlight.html#TrafficLightDomain-setCompleteRedYellowGreenDefinition
            ### getPhases(self) should work on tl_logic but it doesnt, so I have to scrape the phases like the following
            phases = []
            '''
            for phase in tl_logic._phases:
                if ('g' in phase._phaseDef or 'G' in phase._phaseDef) and 'y' not in phase._phaseDef:
                    phases.append(phase._phaseDef)
            '''
            for p in tsc_logic.getPhases():
                # print(p.state)
                if ('g' in p.state or 'G' in p.state) and 'y' not in p.state:
                    phases.append(p.state)

            tsc_data[tsc]['green_phases'] = phases
            tsc_data[tsc]['n_green_phases'] = len(phases)
            tsc_data[tsc]['all_red'] = 'r' * len(phases[0])

            lanes = self.conn.trafficlight.getControlledLanes(tsc)
            ###incoming lanes
            tsc_data[tsc]['inc_lanes'] = list(set(lanes))
            index_to_lane = {i: lane for lane, i in zip(lanes, range(len(lanes)))}
            tsc_data[tsc]['phase_lanes'] = self.action_state_lanes(tsc_data[tsc]['green_phases'], index_to_lane)
            tsc_data[tsc]['inc_edges'] = set([self.conn.lane.getEdgeID(l) for l in lanes])
            phases = sorted(phases)
            ##get one hot for actions and phases for state
            tsc_data[tsc]['action_one_hot'] = self.input_to_one_hot(phases)
            tsc_data[tsc]['int_to_action'] = phases
            tsc_data[tsc]['phase_one_hot'] = self.input_to_one_hot(phases + [tsc_data[tsc]['all_red']])
            tsc_data[tsc]['int_to_phase'] = self.int_to_input(phases + [tsc_data[tsc]['all_red']])
            ###lane lengths for normalization
            tsc_data[tsc]['lane_lengths'] = {l: self.conn.lane.getLength(l) for l in lanes}

        return tsc_data

    def run(self, net_data=None, args=None, exp_replay=None, neural_network=None, eps=None, rl_stats=None):
        ###for batch vehicle data, faster than API calls
        data_constants = [traci.constants.VAR_SPEED, traci.constants.VAR_POSITION, traci.constants.VAR_LANE_ID,
                          traci.constants.VAR_LANE_INDEX]
        self.vehicles = Vehicles(self.conn, data_constants, net_data, self.sim_len, args.demand,
                                 args.scale) if net_data else None

        ###create some intersections
        intersections = [
            Intersection(_id, args.tsc, self.conn, args, net_data['tsc'][_id], exp_replay[_id], neural_network[_id],
                         eps, rl_stats[_id], self.vehicles.get_edge_delay) for _id in
            self.conn.trafficlight.getIDList()] if net_data else []

        print('start running sumo sim on port ' + str(self.port))
        while self.t < self.sim_len:
            ### Điều kiện dừng bổ sung: Dừng khi không còn phương tiện trên đường
            if self.conn.vehicle.getIDCount() == 0:
                print("No vehicles left on the network. Stopping simulation.")
                break

            ###loop thru tsc intersections
            ###run and pass v_data
            lane_vehicles = self.vehicles.run() if self.vehicles else None
            for i in intersections:
                i.run(lane_vehicles)
            self.step()

            ### Đảm bảo self.t được tăng lên
            self.t += 1

        if args and args.mode == 'train':
            ###write travel time mean to csv for graphing after training
            self.write_csv(str(eps) + '.csv',
                           np.mean([self.vehicles.travel_times[v] for v in self.vehicles.travel_times]))

        print('finished running sumo sim on port ' + str(self.port))
        self.cleanup()

    def write_csv(self, fp, data):
        with open(fp, 'a+') as f:
            f.write(str(data) + '\n')

    def step(self):
        self.conn.simulationStep()
        self.t += 1

    def cleanup(self):
        self.conn.close()
        self.sumo_process.terminate()
        print('finished cleaning up sim on port ' + str(self.port) + ' after ' + str(self.t) + ' steps')


if __name__ == '__main__':

    start_t = time.time()
    net_fp = r'C:\Traffic-Light-Scheduling\networks\osm.net.xml.gz'

    # Khởi động mô phỏng SUMO
    port = 9001
    idx = 0
    cfg_fp = r'C:\Traffic-Light-Scheduling\networks\osm.sumocfg'
    sim = SumoSim(port, idx, cfg_fp)
    sim_len = 86400
    nogui = True
    sim.gen_sim(nogui, sim_len)  # SUMO khởi động và kết nối được thiết lập tại đây

    # Sau khi SUMO đã khởi động và kết nối thành công, bạn có thể khởi tạo `NetworkData`
    network_data = NetworkData(net_fp)
    print('network time elapsed ' + str(time.time() - start_t))

    tsc_data = sim.get_tsc_data()
    for tsc in tsc_data:
        print(tsc)
        print(tsc_data[tsc])
    sim.run()



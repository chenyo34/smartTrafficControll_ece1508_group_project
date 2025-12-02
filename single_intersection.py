
import numpy as np
import os 
import matplotlib.pyplot as plt
import time
from sumo_rl import SumoEnvironment
import traci
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci


class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_cmd, tls_id, gui=False,
                 max_steps=3600,
                 c1=1.0,
                 c2=0.3, 
                 c3=0.15,
                 c4=0.05,
                 c5=0.005,
                 noise=False,
                 noise_sigma=1.0
):
        super().__init__()

        cmd_prefix = ["sumo-gui", "--start"] if gui else ["sumo"]
        self.sumo_cmd = cmd_prefix +  sumo_cmd
        self.ts_id = tls_id

        self.steps = 0
        self.max_steps = max_steps
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        self.noise = noise
        self.noise_sigma = noise_sigma

        self._start_sumo()
        self._identify_lanes()

        # Gym Action Space = traffic light phases
        num_phases = len(self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.ts_id)[0].phases)
        self.action_space = spaces.Discrete(num_phases)

        # Gym Observation Space = vehicle count per lane
        num_in = len(self.in_lanes)
        num_out = len(self.out_lanes)
        num_phases = self.action_space.n

        obs_dim = num_in * 3 + num_out + num_phases + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )


        self.prev_phase = self.sumo.trafficlight.getPhase(self.ts_id)
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())
        self.prev_queue = 0


    # ---------------------------------------------------------
    # SUMO START
    # ---------------------------------------------------------
    def _start_sumo(self):
        traci.start(self.sumo_cmd)
        self.sumo = traci


    # ---------------------------------------------------------
    # Auto-detect incoming/outgoing lanes
    # ---------------------------------------------------------
    def _identify_lanes(self):
        self.in_lanes = []
        self.out_lanes = []

        links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        for group in links:
            for incoming, outgoing, _ in group:
                if incoming not in self.in_lanes:
                    self.in_lanes.append(incoming)
                if outgoing not in self.out_lanes:
                    self.out_lanes.append(outgoing)


    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        traci.close(False)
        self._start_sumo()
        self._identify_lanes()

        self.prev_phase = self.sumo.trafficlight.getPhase(self.ts_id)
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())

        obs = self._get_observation()
        return obs, {}


    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def step(self, action):

        self.steps += 1
        # Apply traffic light phase
        self.sumo.trafficlight.setPhase(self.ts_id, int(action))

        # Move simulation by 1 step
        self.sumo.simulationStep()

        # Reward
        reward = self._compute_reward(action)

        # Evaluation Metrics
        avg_speed=self._computer_avg_speed()
        throughput=self._compute_throughput()
        waiting_time=self._compute_avg_waiting_time()
        queue_length = self._compute_queue_length()
        pressure = self._compute_pressure()


        # Update history
        self.prev_phase = action
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())

        # Observation
        obs = self._get_observation()

        # Check collision
        if self.sumo.simulation.getCollidingVehiclesNumber() > 0:
            done = True
        else: 
            done = False
            
        truncated = self.steps >= self.max_steps if hasattr(self, 'max_steps') else False 
        info = {
            "avg_speed": avg_speed,
            "throughput": throughput,
            "waiting_time": waiting_time,
            "queue_length": queue_length,
            "pressure": pressure

        }

        # return obs, reward, done, truncated, info, avg_speed, throughput, waiting_time, queue_length
        return obs, reward, done, truncated, info 

    # ---------------------------------------------------------
    # Compute observation
    # ---------------------------------------------------------
    def _get_observation(self):
        obs = []

        #  Lane-level features
        for lane in self.in_lanes:
            count = self.sumo.lane.getLastStepVehicleNumber(lane)
            queue = self.sumo.lane.getLastStepHaltingNumber(lane)
            speed = self.sumo.lane.getLastStepMeanSpeed(lane)

            # -------- Sensor noise --------
            if hasattr(self, 'noise') and self.noise:
                count = max(0, count + np.random.normal(0, self.noise_sigma))
                queue = max(0, queue + np.random.normal(0, self.noise_sigma))
                speed = max(0, speed + np.random.normal(0, self.noise_sigma))

            obs.extend([count, queue, speed])
        
        for lane in self.out_lanes:
            count = self.sumo.lane.getLastStepVehicleNumber(lane)

            if hasattr(self, 'noise') and self.noise:
                count = max(0, count + np.random.normal(0, self.noise_sigma))

            obs.append(count)

        # Traffic Light phase information
        phase = self.sumo.trafficlight.getPhase(self.ts_id)
        phase_one_hot = np.zeros(self.action_space.n)
        phase_one_hot[phase] = 1
        obs.extend(phase_one_hot.tolist())

        # Pressure
        pressure = self._compute_pressure()
        obs.append(pressure)

        return np.array(obs, dtype=np.float32)


    # ---------------------------------------------------------
    # Reward function
    # ---------------------------------------------------------
    def _compute_reward(self, action):
        queue_reduction = self._compute_queue_reduction()
        queue_abs = self._compute_queue_length()
        pressure = self._compute_pressure()
        switch_penalty = self._compute_switch_penalty(action)
        throughput = self._compute_throughput()
        return (
            self.c1 * queue_reduction
            - self.c2 * queue_abs
            - self.c3 * pressure
            - self.c4 * switch_penalty
            + self.c5 * throughput
        )


    # def _compute_pressure(self):
    #     incoming = sum(self.sumo.lane.getLastStepVehicleNumber(l) for l in self.in_lanes)
    #     outgoing = sum(self.sumo.lane.getLastStepVehicleNumber(l) for l in self.out_lanes)
    #     return incoming - outgoing

    def _compute_pressure(self):
        """
        Computes traffic pressure exactly following:
        P_i = sum_{m in movements(i)} (x_in(m) - x_out(m))
        where each movement m is (incoming_lane -> outgoing_lane).
        """
        pressure = 0
        # All controlled movements of this traffic light
        movements = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        # movements is a list of lists: [[(incoming, outgoing, via)], ...]
        for group in movements:
            for (incoming, outgoing, _) in group:
                x_in = self.sumo.lane.getLastStepVehicleNumber(incoming)
                x_out = self.sumo.lane.getLastStepVehicleNumber(outgoing)
                pressure += (x_in - x_out)
                
        return pressure

    def _compute_switch_penalty(self, action):
        return 1.0 if action != self.prev_phase else 0.0

    def _compute_throughput(self):
        current_passed = len(self.sumo.simulation.getArrivedIDList())
        return max(current_passed - self.prev_passed, 0)
    
    def _compute_queue_length(self):
        total_q = 0
        for lane in self.in_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                if self.sumo.vehicle.getSpeed(v) < 0.1:  # True queue
                    total_q += 1
        return total_q
    
    def _compute_queue_reduction(self):
        Q_t = self._compute_queue_length()
        reduction = self.prev_queue - Q_t
        self.prev_queue = Q_t
        return reduction
    
    def _computer_avg_speed(self):
        vehs_speeds=[]
        for lane in self.in_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                v_speed=self.sumo.vehicle.getSpeed(v)
                vehs_speeds.append(v_speed)
        if len(vehs_speeds) == 0:
            return 0.0
        return np.mean(vehs_speeds)
    
    def _compute_avg_waiting_time(self):
        wts=[]
        
        for lane in self.in_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                wt = self.sumo.vehicle.getWaitingTime(v)  # 或 getAccumulatedWaitingTime(v)
                wts.append(wt)
        if len(wts)==0:
            return 0.0
        return np.mean(wts)

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------
    def close(self):
        traci.close()


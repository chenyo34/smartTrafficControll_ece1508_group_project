
from pickle import FALSE
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
                 c4=0.01,
                 c5=0.005,
                 noise=False,
                 noise_sigma=1.0
):
        super().__init__()

        cmd_prefix = ["sumo-gui", "--start"] if gui else ["sumo"]
        # Add flags to suppress SUMO output/warnings (only if not already present)
        quiet_flags = []
        existing_flags = set(sumo_cmd)
        # Check for various forms of the flags
        if "--no-step-log" not in existing_flags and "-S" not in existing_flags:
            quiet_flags.append("--no-step-log")
        if "--no-warnings" not in existing_flags and "-W" not in existing_flags and "--suppress-warnings" not in existing_flags:
            quiet_flags.append("--no-warnings")
        if "--no-duration-log" not in existing_flags:
            quiet_flags.append("--no-duration-log")
        self.sumo_cmd = cmd_prefix + quiet_flags + sumo_cmd
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

        self.green_phases = [0, 2, 4, 6]
        self.yellow_phases = [1, 3, 5, 7]

        # Override action space to 4 (directions, not SUMO phases)
        self.action_space = spaces.Discrete(4)

        # Gym Observation Space = vehicle count per lane
        # Calculate AFTER setting action_space to 4, since _get_observation() uses action_space.n
        num_in = len(self.in_lanes)
        num_out = len(self.out_lanes)
        num_phases = self.action_space.n  # This is now 4, not 8

        obs_dim = num_in * 3 + num_out + num_phases + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize prev_phase as action (direction index 0-3), not SUMO phase
        initial_phase = self.sumo.trafficlight.getPhase(self.ts_id)
        if initial_phase in self.green_phases:
            self.prev_phase = self.green_phases.index(initial_phase)
        elif initial_phase in self.yellow_phases:
            self.prev_phase = self.yellow_phases.index(initial_phase)
        else:
            self.prev_phase = 0  # Default to direction 0
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())
        self.prev_queue = 0

        self.fixed_duration = 2

        # ---------------------------------------------------------
        # Phase switch helper function
        # ---------------------------------------------------------

    def _next_phase_with_constraint(self, action: int):

        cur_phase = self.sumo.trafficlight.getPhase(self.ts_id)

        #  phase
        if cur_phase in self.green_phases:
            cur_dir = self.green_phases.index(cur_phase)  

            
            if action == cur_dir:
                next_phase = cur_phase
            else:
                next_phase = self.yellow_phases[cur_dir]

            return next_phase

        elif cur_phase in self.yellow_phases:
            cur_dir = self.yellow_phases.index(cur_phase)  

            desired_dir = action

            if desired_dir == cur_dir:
                # 随机选择一个可行的phase（排除当前方向）
                feasible_dirs = [i for i in range(4) if i != cur_dir]
                desired_dir = np.random.choice(feasible_dirs)

            next_phase = self.green_phases[desired_dir]
            return next_phase

        else:
            return self.green_phases[0]

            

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

        # Initialize prev_phase as action (direction index 0-3), not SUMO phase
        initial_phase = self.sumo.trafficlight.getPhase(self.ts_id)
        if initial_phase in self.green_phases:
            self.prev_phase = self.green_phases.index(initial_phase)
        elif initial_phase in self.yellow_phases:
            self.prev_phase = self.yellow_phases.index(initial_phase)
        else:
            self.prev_phase = 0  # Default to direction 0
        self.prev_queue = 0
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())

        obs = self._get_observation()
        return obs, {}


    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    # def step_wrong(self, action):

    #     self.steps += 1
    #     # Apply traffic light phase
    #     self.sumo.trafficlight.setPhase(self.ts_id, int(action))

    #     # Move simulation by 1 step
    #     self.sumo.simulationStep()

    #     # Reward
    #     reward = self._compute_reward(action)

    #     # Evaluation Metrics
    #     avg_speed=self._computer_avg_speed()
    #     throughput=self._compute_throughput()
    #     waiting_time=self._compute_avg_waiting_time()
    #     queue_length = self._compute_queue_length()
    #     pressure = self._compute_pressure()


    #     # Update history
    #     self.prev_phase = action
    #     self.prev_passed = len(self.sumo.simulation.getArrivedIDList())

    #     # Observation
    #     obs = self._get_observation()

    #     # Check collision
    #     if self.sumo.simulation.getCollidingVehiclesNumber() > 0:
    #         done = True
    #     else: 
    #         done = False
            
    #     truncated = self.steps >= self.max_steps if hasattr(self, 'max_steps') else False 
    #     info = {
    #         "avg_speed": avg_speed,
    #         "throughput": throughput,
    #         "waiting_time": waiting_time,
    #         "queue_length": queue_length,
    #         "pressure": pressure

    #     }

    #     # return obs, reward, done, truncated, info, avg_speed, throughput, waiting_time, queue_length
    #     return obs, reward, done, truncated, info 
    def step(self, action):
        self.steps += 1

        # 1. 根据当前 phase + PPO action，算出“合法的下一相位”
        next_phase = self._next_phase_with_constraint(int(action))

        # 2. 设置相位
        self.sumo.trafficlight.setPhase(self.ts_id, next_phase)

        total_reward = 0.0
        done = False
        truncated = False

        # 3. 固定 duration = 2 秒：在同一个相位下，连跑 2 个 SUMO step
        for _ in range(self.fixed_duration):
            self.sumo.simulationStep()

            r = self._compute_reward(next_phase)   # 你原来怎么算就怎么算
            total_reward += r

            # 如果你有碰撞终止条件就放这里
            if self.sumo.simulation.getCollidingVehiclesNumber() > 0:
                done = True
            else: 
                done = False

            truncated = self.steps >= self.max_steps if hasattr(self, 'max_steps') else False 

        # Evaluation Metrics
        avg_speed=self._computer_avg_speed()
        throughput=self._compute_throughput()
        waiting_time=self._compute_avg_waiting_time()
        queue_length = self._compute_queue_length()
        pressure = self._compute_pressure()       
                # Update history
        self.prev_phase = action
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())
    
        # 4. 取最后一个时刻的观测
        obs = self._get_observation()

        info = {
            "avg_speed": avg_speed,
            "throughput": throughput,
            "waiting_time": waiting_time,
            "queue_length": queue_length,
            "pressure": pressure

        }

        return obs, total_reward, done, truncated, info
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
        # Map SUMO phase (0-7) to direction index (0-3) for one-hot encoding
        phase = self.sumo.trafficlight.getPhase(self.ts_id)
        phase_one_hot = np.zeros(self.action_space.n)
        
        # Convert SUMO phase to direction index
        if phase in self.green_phases:
            dir_idx = self.green_phases.index(phase)
        elif phase in self.yellow_phases:
            dir_idx = self.yellow_phases.index(phase)
        else:
            dir_idx = 0  # Default to direction 0
        
        phase_one_hot[dir_idx] = 1
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
        return (self.c1 * queue_reduction
                    + self.c5 * throughput
                    - self.c2 * queue_abs
                    - self.c3 * pressure
                    - self.c4 * switch_penalty
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
        throughput = max(current_passed - self.prev_passed, 0)
        norm_throughput = 0.5 * (np.tanh(throughput / 3) + 1)
        return norm_throughput
    
    def _compute_queue_length(self):
        total_q = 0
        for lane in self.in_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                if self.sumo.vehicle.getSpeed(v) < 0.1:  # True queue
                    total_q += 1
        norm_total_q = total_q / 80
        return norm_total_q
    
    def _compute_queue_reduction(self):
        Q_t = self._compute_queue_length() 
        reduction = self.prev_queue - Q_t
        self.prev_queue = Q_t
        norm_reduction = reduction / 80        
        return norm_reduction
    
    def _computer_avg_speed(self):
        vehs_speeds=[]
        for lane in self.in_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                v_speed=self.sumo.vehicle.getSpeed(v)
                vehs_speeds.append(v_speed)
        if len(vehs_speeds) == 0:
            return 0.0
        norm_vehs_speed = np.mean(vehs_speeds) / 13.9
        return norm_vehs_speed
    
    def _compute_avg_waiting_time(self):
        wts=[]
        
        for lane in self.in_lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                wt = self.sumo.vehicle.getWaitingTime(v)  # 或 getAccumulatedWaitingTime(v)
                wts.append(wt)
        if len(wts)==0:
            return 0.0
        waiting_time = np.mean(wts)
        norm_wt = 0.5 * (np.tanh(waiting_time / 30) + 1)
        return norm_wt

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------
    def close(self):
        traci.close()



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
                 alpha=1.0,
                 beta=0.5, 
                 gamma=1.0):
        super().__init__()

        cmd_prefix = ["sumo-gui", "--start"] if gui else ["sumo"]
        self.sumo_cmd = cmd_prefix +  sumo_cmd
        self.ts_id = tls_id

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self._start_sumo()
        self._identify_lanes()

        # Gym Action Space = traffic light phases
        num_phases = len(self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.ts_id)[0].phases)
        self.action_space = spaces.Discrete(num_phases)

        # Gym Observation Space = vehicle count per lane
        obs_dim = len(self.in_lanes) + len(self.out_lanes)
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(obs_dim,), dtype=np.float32
        )

        self.prev_phase = self.sumo.trafficlight.getPhase(self.ts_id)
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())


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
        # Apply traffic light phase
        self.sumo.trafficlight.setPhase(self.ts_id, int(action))

        # Move simulation by 1 step
        self.sumo.simulationStep()

        # Reward
        reward = self._compute_reward(action)

        # Update history
        self.prev_phase = action
        self.prev_passed = len(self.sumo.simulation.getArrivedIDList())

        # Observation
        obs = self._get_observation()

        done = False
        truncated = False
        info = {}

        return obs, reward, done, truncated, info


    # ---------------------------------------------------------
    # Compute observation
    # ---------------------------------------------------------
    def _get_observation(self):
        lane_counts = []
        for lane in self.in_lanes + self.out_lanes:
            lane_counts.append(
                self.sumo.lane.getLastStepVehicleNumber(lane)
            )
        return np.array(lane_counts, dtype=np.float32)


    # ---------------------------------------------------------
    # Reward function
    # ---------------------------------------------------------
    def _compute_reward(self, action):
        return (
            - self.alpha * self._compute_pressure()
            - self.beta * self._compute_switch_penalty(action)
            + self.gamma * self._compute_throughput()
        )


    def _compute_pressure(self):
        incoming = sum(self.sumo.lane.getLastStepVehicleNumber(l) for l in self.in_lanes)
        outgoing = sum(self.sumo.lane.getLastStepVehicleNumber(l) for l in self.out_lanes)
        return incoming - outgoing


    def _compute_switch_penalty(self, action):
        return 1.0 if action != self.prev_phase else 0.0


    def _compute_throughput(self):
        current_passed = len(self.sumo.simulation.getArrivedIDList())
        return max(current_passed - self.prev_passed, 0)


    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------
    def close(self):
        traci.close()


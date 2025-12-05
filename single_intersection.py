import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import sys


class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_cmd, tls_id, gui=False,
                 max_steps=3600,
                 c1=1.0, c2=1.0, c3=0.0, c4=0.1, c5=0.2,
                 noise=False, noise_sigma=1.0):
        super().__init__()

        # === 修复点：自动添加 sumo 可执行文件名 ===
        # 你的 sumo_cmd 变量里只有 flags (如 --no-warnings)，没有 'sumo'
        # 所以这里必须手动把 'sumo' 加到列表最前面
        if gui:
            binary = ["sumo-gui", "--start", "--quit-on-end"]
        else:
            binary = ["sumo"]

        # 使用 list() 复制一份，防止修改原始变量
        self.sumo_cmd = binary + list(sumo_cmd)

        self.ts_id = tls_id
        self.max_steps = max_steps
        self.steps = 0

        # 奖励系数
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        self.noise = noise
        self.noise_sigma = noise_sigma

        # 控制参数
        self.min_green = 5  # 最小绿灯时间
        self.yellow_time = 3  # 黄灯时间
        self.fixed_duration = 15  # 每次动作持续时间

        # 初始化SUMO
        self._start_sumo()
        self._identify_lanes()

        # Action Space (4个相位)
        self.action_space = spaces.Discrete(4)
        self.green_phases = [0, 2, 4, 6]
        self.yellow_phases = [1, 3, 5, 7]

        # 建立 Action -> Target Green Phase 的映射字典
        self.action_to_phase = {
            0: 0,  # Action 0 -> Phase 0
            1: 2,  # Action 1 -> Phase 2
            2: 4,  # Action 2 -> Phase 4
            3: 6  # Action 3 -> Phase 6
        }

        # 初始化历史变量
        self._init_history_vars()

        # === 自动检测 Observation 维度 ===
        dummy_obs = self._get_observation()
        obs_dim = len(dummy_obs)
        print(f"[Env Init] Detected Observation Dimension: {obs_dim}")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        try:
            traci.close()
        except:
            pass

        self._start_sumo()
        self._identify_lanes()
        self._init_history_vars()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        action = int(action)
        current_time = self.sumo.simulation.getTime()

        # === 1. 最小绿灯保护逻辑 ===
        cur_phase_sumo = self.sumo.trafficlight.getPhase(self.ts_id)
        if cur_phase_sumo in self.green_phases:
            cur_action_idx = self.green_phases.index(cur_phase_sumo)
            if action != cur_action_idx:
                if (current_time - self.last_switch_time) < self.min_green:
                    action = cur_action_idx

        # === 2. 更新动作历史 ===
        if action != self.last_action:
            self.last_switch_time = current_time
            self.last_action = action

        # === 3. 设置相位 ===
        next_phase = self._next_phase_with_constraint(action)
        self.sumo.trafficlight.setPhase(self.ts_id, next_phase)

        # === 4. 确定持续时间 ===
        if next_phase in self.yellow_phases:
            run_duration = self.yellow_time
        else:
            run_duration = self.fixed_duration

        # === 5. 执行仿真循环 (关键修改！) ===
        done = False
        current_step_throughput = 0  # <--- 初始化累加器

        for _ in range(run_duration):
            self.sumo.simulationStep()

            # 【修复点】在循环里每一帧都统计离开的车
            current_step_throughput += self.sumo.simulation.getArrivedNumber()

            if self.sumo.simulation.getCollidingVehiclesNumber() > 0:
                done = True
                break

        # === 6. 计算奖励 ===
        # 把刚才累加的吞吐量传给 reward 计算函数，或者存入 self 供调用
        self.last_step_throughput = current_step_throughput  # 存起来给 info 用
        total_reward = self._compute_reward(action, current_step_throughput)  # 传参进去

        # === 7. 检查超时 ===
        truncated = self.steps >= self.max_steps

        # === 8. 收集信息 ===
        self.prev_phase = action
        obs = self._get_observation()

        info = {
            "avg_speed": self._computer_avg_speed(),
            "throughput": current_step_throughput,  # 这里直接用累加的值
            "waiting_time": self._compute_avg_waiting_time(),
            "queue_length": self._compute_queue_length(),
            "pressure": self._compute_pressure(),
            "action": action
        }

        return obs, total_reward, done, truncated, info
    def _get_observation(self):
        obs = []
        # 归一化常数
        MAX_VEH = 20.0
        MAX_SPEED = 15.0

        # 入道特征
        for lane in self.in_lanes:
            n = self.sumo.lane.getLastStepVehicleNumber(lane)
            q = self.sumo.lane.getLastStepHaltingNumber(lane)
            s = self.sumo.lane.getLastStepMeanSpeed(lane)

            # 必须归一化！
            obs.extend([
                min(1.0, n / MAX_VEH),
                min(1.0, q / MAX_VEH),
                min(1.0, s / MAX_SPEED)
            ])

        # 出道特征
        for lane in self.out_lanes:
            n = self.sumo.lane.getLastStepVehicleNumber(lane)
            obs.append(min(1.0, n / MAX_VEH))

        # 相位 One-Hot (5位: 4个方向 + 1个黄灯)
        phase_oh = [0] * 5
        phase = self.sumo.trafficlight.getPhase(self.ts_id)
        if phase in self.green_phases:
            phase_oh[self.green_phases.index(phase)] = 1
        elif phase in self.yellow_phases:
            phase_oh[4] = 1  # Yellow bit
        else:
            phase_oh[4] = 1  # Fallback

        obs.extend(phase_oh)

        # Pressure
        obs.append(self._compute_pressure() / 10.0)  # 简单缩放

        return np.array(obs, dtype=np.float32)

        # 修改这里，增加 throughput_count 参数
    def _compute_reward(self, action, throughput_count):
            pressure = self._compute_pressure()
            queue = self._compute_queue_length()
            switch = 1.0 if action != self.prev_phase else 0.0

            # 直接使用传进来的数值，不要再调用那个错误的函数了
            throughput = throughput_count

            # R = - Queue - Switch + Throughput
            reward = - (self.c2 * queue) - (self.c4 * switch) + (self.c5 * throughput)
            reward -= 0.05 * pressure

            return reward

    # --- 辅助函数 ---
    def _start_sumo(self):
        traci.start(self.sumo_cmd)
        self.sumo = traci

    def _init_history_vars(self):
        self.prev_passed = 0
        self.last_switch_time = 0
        self.last_action = 0
        self.prev_phase = 0
        self.prev_queue = self._compute_queue_length()  # 正确初始化

    def _identify_lanes(self):
        self.in_lanes = [l for l in self.sumo.trafficlight.getControlledLanes(self.ts_id) if ":t" not in l]
        # 简单处理：假设所有非 in_lanes 的连接路是 out (根据你的net.xml实际情况可能需要调整)
        self.out_lanes = []
        # 这里简化处理，为了不报错，先沿用你之前的逻辑或者只用 in_lanes
        # 如果你之前的 _identify_lanes 是对的，请保留你的逻辑。
        # 这里我写一个通用的：
        links = self.sumo.trafficlight.getControlledLinks(self.ts_id)
        for group in links:
            for i, o, _ in group:
                if i not in self.in_lanes: self.in_lanes.append(i)
                if o not in self.out_lanes: self.out_lanes.append(o)
        self.in_lanes = sorted(list(set(self.in_lanes)))
        self.out_lanes = sorted(list(set(self.out_lanes)))

    def _next_phase_with_constraint(self, action):
        """
        决定下一时刻的交通灯相位。
        逻辑：
        1. 获取 Agent 想要的目标绿灯相位 (target_phase)。
        2. 检查当前相位 (current_phase)。
        3. 如果 当前是绿灯 且 当前 != 目标 -> 变黄灯。
        4. 如果 当前是黄灯 -> 直接变目标绿灯 (因为step里已经跑完黄灯时间了)。
        """

        # 1. 解码动作：Action(0-3) -> Target Phase(0,2,4,6)
        target_phase = self.action_to_phase[int(action)]

        # 2. 获取当前 SUMO 里的真实相位 (0-7)
        current_phase = self.sumo.trafficlight.getPhase(self.ts_id)

        # === 情况 A: 当前是绿灯 ===
        if current_phase in self.green_phases:
            # 如果现在的绿灯就是 Agent 想要的 -> 保持不变
            if current_phase == target_phase:
                return current_phase
            else:
                # 如果不一样，说明要切灯。
                # 绿灯不能直接跳到另一个绿灯，必须先变对应的黄灯。
                # 在你的 net.xml 里，黄灯ID 永远是 绿灯ID + 1
                # 例如: Phase 0(G) -> Phase 1(y); Phase 2(G) -> Phase 3(y)
                return current_phase + 1

        # === 情况 B: 当前是黄灯 ===
        elif current_phase in self.yellow_phases:
            # 黄灯时间在 step() 函数的仿真循环里已经跑完了 (3秒)。
            # 所以这里直接切换到 Agent 指定的 目标绿灯 即可。
            # 哪怕黄灯对应的并不是这个绿灯 (比如 Phase 1 黄灯后想去 Phase 4)，
            # 这里的逻辑也强制让它切过去，达成 Agent 的意图。
            return target_phase

        # === 情况 C: 异常/全红 (Fallback) ===
        else:
            return 0

    def _compute_pressure(self):
        p = 0
        for l in self.in_lanes:
            p += self.sumo.lane.getLastStepVehicleNumber(l)
        for l in self.out_lanes:
            p -= self.sumo.lane.getLastStepVehicleNumber(l)
        return abs(p)

    def _compute_queue_length(self):
        q = 0
        for l in self.in_lanes:
            q += self.sumo.lane.getLastStepHaltingNumber(l)
        return q / 40.0  # 归一化

    def _compute_throughput(self):
        curr = len(self.sumo.simulation.getArrivedIDList())
        diff = curr - self.prev_passed
        return diff  # 返回通过的车辆数

    def _compute_avg_waiting_time(self):
        # 简单实现
        total_wait = 0
        count = 0
        for l in self.in_lanes:
            total_wait += self.sumo.lane.getWaitingTime(l)
            count += self.sumo.lane.getLastStepVehicleNumber(l)
        return (total_wait / max(1, count)) / 100.0  # 归一化

    def _computer_avg_speed(self):
        s = 0
        for l in self.in_lanes:
            s += self.sumo.lane.getLastStepMeanSpeed(l)
        return (s / len(self.in_lanes)) / 13.89

    def close(self):
        traci.close()
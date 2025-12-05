

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from single_intersection import TrafficEnv



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_size = 64

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [batch, obs_dim]
        返回:
          logits: [batch, act_dim]
          value:  [batch]
        """
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, obs):
        """
        输入单步 obs (numpy array)，返回:
          action: int
          log_prob: float
          value: float
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.forward(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return (
            action.item(),
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def evaluate_actions(self, obs_batch, act_batch):
        """
        在训练时用:
          obs_batch: [batch, obs_dim]
          act_batch: [batch]
        返回:
          log_probs: [batch]
          entropy:   [batch]
          values:    [batch]
        """
        logits, values = self.forward(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act_batch)
        entropy = dist.entropy()
        return log_probs, entropy, values
    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = self.forward(obs)
        return value.cpu().item()



def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = np.zeros(n)
    last_adv = 0

    for t in reversed(range(n)):
        mask = 1.0 - dones[t]  # ❗ only REAL done cuts trajectory
        delta = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
        next_value = values[t]

    returns = advantages + values
    return advantages, returns



def collect_rollout(env, model, n_steps):

    obs_list = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    values = []

    avg_speeds = []
    throughputs = []
    waiting_times = []
    queue_length = []
    pressures = []

    obs, info = env.reset()

    for step in range(n_steps):
        # --- model step ---
        action, log_prob, value = model.act(obs)

        # --- env step ---
        next_obs, reward, done, truncated, info = env.step(action)

        # --- store ---
        obs_list.append(obs)
        actions.append(action)
        logprobs.append(log_prob)
        rewards.append(reward)

        # 🚨 关键：GAE mask 只在 done=True 时为 1
        # truncated 只是时间到，不代表 episode 结束
        dones.append(float(done))

        values.append(value)

        avg_speeds.append(info["avg_speed"])
        throughputs.append(info["throughput"])
        waiting_times.append(info["waiting_time"])
        queue_length.append(info["queue_length"])
        pressures.append(info["pressure"])

        # [FIXED] 必须处理 truncated，否则环境永远不重置
        dones.append(float(done or truncated))
        if done or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    # ---- rollout 结束后：bootstrap V(s_T) ----
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, next_value_t = model.forward(obs_t)
    next_value = next_value_t.cpu().item()

    batch = {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "logprobs": np.array(logprobs, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        # 🚨 dones 不包含 truncated！！
        "dones": np.array(dones, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),

        "next_obs": obs,
        "next_value": next_value,

        "avg_speeds": np.array(avg_speeds, dtype=np.float32),
        "throughputs": np.array(throughputs, dtype=np.float32),
        "waiting_times": np.array(waiting_times, dtype=np.float32),
        "queue_length": np.array(queue_length, dtype=np.float32),
        "pressures": np.array(pressures, dtype=np.float32)
    }

    return batch





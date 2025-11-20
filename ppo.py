

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
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
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




def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    rewards: [T]
    values:  [T]
    dones:   [T]
    next_value: 标量 (V(s_{T}))
    返回:
      advantages: [T]
      returns:    [T] = advantages + values
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # done=True 时不再 bootstrap
        delta = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = last_adv = delta + gamma * lam * mask * last_adv
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def collect_rollout(env, model, n_steps):
    """
    从当前策略 π_θ 下采样 n_steps 步数据 (可能跨多个 episode)。
    返回一个字典，包含:
        obs, actions, logprobs, rewards, dones, values, next_obs, next_value
    """

    obs_list = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    values = []

    avg_speeds=[]
    throughputs=[]
    waiting_times=[]

    obs, info = env.reset()

    for step in range(n_steps):
        action, log_prob, value = model.act(obs)

        next_obs, reward, done, truncated, info, avg_speed, throughput, waiting_time = env.step(action)
        # 这个 env 的 done 一直是 False，目前可以忽略 truncated，按持续任务处理

        obs_list.append(obs)
        actions.append(action)
        logprobs.append(log_prob)
        rewards.append(reward)
        dones.append(float(done or truncated))
        values.append(value)

        avg_speeds.append(avg_speed)
        throughputs.append(throughput)
        waiting_times.append(waiting_time)

        obs = next_obs

        if done or truncated :
            obs, info = env.reset()
        
        

    # rollout 结束后，用最后一个状态估计 V(s_T)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, next_value_t = model.forward(obs_t)
    next_value = next_value_t.cpu().item()

    batch = {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "logprobs": np.array(logprobs, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "next_obs": obs,
        "next_value": next_value,
        "avg_speeds": np.array(avg_speeds, dtype=np.float32),
        "throughputs": np.array(throughputs, dtype=np.float32),
        "waiting_times": np.array(waiting_times, dtype=np.float32)
    }

    return batch




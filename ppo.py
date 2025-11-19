

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from single_intersection import TrafficEnv  # 你的环境文件


GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

N_STEPS = 2048        # 每次采样的总步数
N_EPOCHS = 10         # 每批样本上更新多少个 epoch
MINI_BATCH_SIZE = 256 # 从 N_STEPS 里分成小批次
TOTAL_TIMESTEPS = 200_000  # 总训练步数（可改）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
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




def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
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

    obs, info = env.reset()

    for step in range(n_steps):
        action, log_prob, value = model.act(obs)

        next_obs, reward, done, truncated, info = env.step(action)
        # 这个 env 的 done 一直是 False，目前可以忽略 truncated，按持续任务处理

        obs_list.append(obs)
        actions.append(action)
        logprobs.append(log_prob)
        rewards.append(reward)
        dones.append(float(done or truncated))
        values.append(value)

        obs = next_obs

        if done or truncated :
            obs, info = env.reset()
        
        

    # rollout 结束后，用最后一个状态估计 V(s_T)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
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
    }

    return batch



def train_ppo(
    env=None,
    GAMMA = 0.99,
    GAE_LAMBDA = 0.95,
    CLIP_EPS = 0.2,
    LR = 3e-4,
    ENT_COEF = 0.01,
    VF_COEF = 0.5,
    MAX_GRAD_NORM = 0.5,
    N_STEPS = 2048, 
    N_EPOCHS = 10,        
    MINI_BATCH_SIZE = 256, 
    TOTAL_TIMESTEPS = 200_000  
):
    if env is None:
        raise ValueError("Please provide a valid environment instance.")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print("Observation dim:", obs_dim)
    print("Action dim:", act_dim)

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    global_step = 0
    episode_returns = []
    current_ep_return = 0.0

    obs, info = env.reset()

    while global_step < TOTAL_TIMESTEPS:
        # 1) 采样一批 rollout
        batch = collect_rollout(env, model, N_STEPS)
        global_step += N_STEPS

        obs_arr = batch["obs"] 
        actions_arr = batch["actions"]
        old_logprobs_arr = batch["logprobs"]
        rewards_arr = batch["rewards"]
        dones_arr = batch["dones"]
        values_arr = batch["values"]
        next_value = batch["next_value"]

        # 2) 计算 GAE & returns
        advantages, returns = compute_gae(
            rewards_arr, values_arr, dones_arr, next_value,
            gamma=GAMMA, lam=GAE_LAMBDA
        )

        # 标准化 advantage 可以加速训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转成 tensor
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions_arr, dtype=torch.int64, device=DEVICE)
        old_logprobs_t = torch.tensor(old_logprobs_arr, dtype=torch.float32, device=DEVICE)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # 3) PPO 多 epoch 训练
        dataset_size = N_STEPS
        indices = np.arange(dataset_size)

        for epoch in range(N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logprobs = old_logprobs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # 计算当前策略下 log_prob, entropy, value
                new_logprobs, entropy, values_pred = model.evaluate_actions(mb_obs, mb_actions)

                # ratio = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # PPO clipped surrogate
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # value function loss
                critic_loss = nn.MSELoss()(values_pred, mb_returns)

                # entropy 促进探索
                entropy_loss = -entropy.mean()

                loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()



        # 4) 打印训练信息（粗略统计一下最近一批的平均奖励）
        batch_mean_return = rewards_arr.mean() * N_STEPS  # 很粗略，仅为参考
        print(f"[Step {global_step}] recent mean reward per step: {rewards_arr.mean():.3f}")

    env.close()
    torch.save(model.state_dict(), "ppo_traffic_signal.pth")
    print("Training finished, model saved to ppo_traffic_signal.pth")


if __name__ == "__main__":
    train_ppo()

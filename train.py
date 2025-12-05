from single_intersection import TrafficEnv
import numpy as np
import os 
import matplotlib.pyplot as plt
import time
from sumo_rl import SumoEnvironment
import torch
import torch.optim as optim
from torch import nn
from ppo import ActorCritic, compute_gae, collect_rollout

from helper_func import plot_traff_metrics

def train_ppo(
    model,
    env=None,
    GAMMA = 0.99,
    GAE_LAMBDA = 0.95,
    CLIP_EPS = 0.2,
    LR = 3e-3,
    ENT_COEF = 0.01,
    VF_COEF = 0.5,
    MAX_GRAD_NORM = 0.5,
    N_STEPS = 256, 
    N_EPOCHS = 10,        
    MINI_BATCH_SIZE = 64, 
    TOTAL_TIMESTEPS = 10_000,
    close_env = True,
    save_model = True,
    model_save_path = "ppo_traffic_signal.pth",
):
    

    if env is None:
        raise ValueError("Please provide a valid environment instance.")
    elif model is None:
        raise ValueError("Please provide a valid model instance.")
    
    # obs_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.n
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # print("Observation dim:", obs_dim)
    # print("Action dim:", act_dim)

    global_step = 0
    # episode_returns = []
    # current_ep_return = 0.0

    obs, info = env.reset()

    appended_rewards=[]
    appended_avg_speeds=[]
    appended_throughputs=[]
    appended_waiting_times=[]

    while global_step < TOTAL_TIMESTEPS:

        # Adding a debug output to reveal the steps
        if global_step % 100 == 0:
            print(f"Debug: Reached {global_step} steps")
        
        # 1) Global Step: Roll-out sampling
        batch = collect_rollout(env, model, N_STEPS)
        global_step += N_STEPS

        obs_arr = batch["obs"] 
        actions_arr = batch["actions"]
        old_logprobs_arr = batch["logprobs"]
        rewards_arr = batch["rewards"]
        dones_arr = batch["dones"]
        values_arr = batch["values"]
        next_value = batch["next_value"]

        # 2) Global Step:  Extract the main metric result from the batch
        avg_speeds = batch["avg_speeds"]
        throughputs = batch["throughputs"]
        waiting_times = batch["waiting_times"]
        queue_lengths = batch["queue_length"]

        # 3)  Global Step: Calcuate the advantages and the returns
        advantages, returns = compute_gae(
            rewards_arr, values_arr, 
            dones_arr, next_value,
            gamma=GAMMA, lam=GAE_LAMBDA
        )

        #  Global Step: Standarize the advantage for better and faster convergence
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        #  Global Step: Tensor type conversion 
        obs_t = torch.tensor(obs_arr, dtype=torch.float32)
        actions_t = torch.tensor(actions_arr, dtype=torch.int64)
        old_logprobs_t = torch.tensor(old_logprobs_arr, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Multiple epochs of updates
        dataset_size = N_STEPS
        indices = np.arange(dataset_size)

        for _ in range(N_EPOCHS):
            # Print debug info 

            np.random.shuffle(indices)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logprobs = old_logprobs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Compute log_prob, entropy, value
                new_logprobs, entropy, values_pred = model.evaluate_actions(mb_obs, mb_actions)

                # ratio = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # PPO clipped surrogate
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # value function loss
                critic_loss = nn.MSELoss()(values_pred, mb_returns)

                # entropy for exploring 
                entropy_loss = -entropy.mean()

                loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()



        # 4) Compute the result for the current batch for metric measurement 
        batch_rewards_return = rewards_arr.mean()
        batch_speeds_return = avg_speeds.mean()
        batch_throughputs_return = throughputs.mean()
        batch_waiting_times_return = waiting_times.mean()
        batch_queue_length_return = queue_lengths.mean()

        appended_rewards.append(batch_rewards_return)
        appended_avg_speeds.append(batch_speeds_return)
        appended_throughputs.append(batch_throughputs_return)
        appended_waiting_times.append(batch_waiting_times_return)

        # Debug printout: key metrics during training
        progress = (global_step / TOTAL_TIMESTEPS) * 100
        print(
            f"[Training] Step {global_step}/{TOTAL_TIMESTEPS} ({progress:.1f}%) | "
            f"Reward: {batch_rewards_return:.3f} | "
            f"Waiting time: {batch_waiting_times_return:.3f} | "
            f"Queue length: {batch_queue_length_return:.3f} | "
            f"Throughput: {batch_throughputs_return:.3f}"
        )

    if close_env:
        env.close()
    
    if save_model:
        torch.save(model.state_dict(), model_save_path)
        print(f"Training finished, model saved to {model_save_path}")
    else:
        print("Training finished")

    # Plotting the eval. graphs
    plot_traff_metrics(
        appended_rewards,
        appended_avg_speeds,
        appended_throughputs,
        appended_waiting_times
    )

    # return appended_rewards, appended_avg_speeds, appended_throughputs, appended_waiting_times

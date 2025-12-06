import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import time


def plot_traff_metrics(
        rewards,
        avg_speeds, 
        throughputs,
        waiting_times,
        title="Traffic Metrics Over Episods",
        save_path=None):
    
    steps = np.arange(len(rewards))

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Create the paths for saving plots
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,2)

    # Rewards Subplot
    plt.subplot(2,2,1)
    plt.plot(steps,rewards,label='Reward',color='orange')
    plt.title("Reward Over Episodes")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)

    # Average Reward Subplot
    plt.subplot(2,2,2)
    plt.plot(steps,avg_speeds,label='AvgSpeedward',color='blue')
    plt.title("AvgSpeed Over Episodes")
    plt.xlabel("Step")
    plt.ylabel("AvgSpeed")
    plt.grid(True)
    
    # Throughput Subplot
    plt.subplot(2,2,3)
    plt.plot(steps,throughputs,label='Throughputs',color='green')
    plt.title("Throughputs Over Episodes")
    plt.xlabel("Step")
    plt.ylabel("Throughputs")
    plt.grid(True)

    # Waiting Time Subplot
    plt.subplot(2,2,4)
    plt.plot(steps,waiting_times,label='Waiting-Time',color='red')
    plt.title("Waiting-Time Over Episodes")
    plt.xlabel("Step")
    plt.ylabel("Waiting-Time")
    plt.grid(True)
    
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    import csv

# Tune hyperparameters
# def tune_hyperparameters(configs):
#     best_config = None
#     best_reward = -np.inf

#     for lr in configs["LR"]:
#         for n_steps in configs["N_STEPS"]:
#             for n_epochs in configs["N_EPOCHS"]:
#                 for mini_batch_size in configs["MINI_BATCH_SIZE"]:
#                     for gamma  in configs["gamma"]:
#                         for clip_range in configs["clip_range"]:
#                             print(f"Testing configuration: LR={lr}, N_STEPS={n_steps}, N_EPOCHS={n_epochs}, MINI_BATCH_SIZE={mini_batch_size}, GAMMA={gamma}, CLIP_RANGE={clip_range}")
#                             env = TrafficEnv(
#                                 sumo_cmd=sumo_cmd,
#                                 tls_id=TLS_ID,
#                                 gui=False   # Render the environment to set it to True
#                             )
#                             # model = ActorCritic(
#                             #     env=env,
#                             #     learning_rate=lr,
#                             #     n_steps=n_steps,
#                             #     n_epochs=n_epochs,
#                             #     mini_batch_size=mini_batch_size,
#                             #     gamma=gamma,
#                             #     clip_range=clip_range
#                             # )
#                             obs_dim = env.observation_space.shape[0]
#                             act_dim = env.action_space.n
#                             model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)

                            
#                             # Train the model
#                             train_ppo(
#                                 env,
#                                 model,
#                                 n_steps=n_steps,
#                                 n_epochs=n_epochs,
#                                 mini_batch_size=mini_batch_size,
#                                 gamma=gamma,
#                                 clip_range=clip_range,
#                                 learning_rate=lr
#                             )
#                             results = evaluate_agent(model, env, agent = "rl", n_eval_episodes=10)
#                             mean_reward = results["avg_reward"] 
#                             print(f"Mean Reward: {mean_reward}")

#                             if mean_reward > best_reward:
#                                 best_reward = mean_reward
#                                 best_config = {
#                                     "LR": lr,
#                                     "N_STEPS": n_steps,
#                                     "N_EPOCHS": n_epochs,
#                                     "MINI_BATCH_SIZE": mini_batch_size,
#                                     "GAMMA": gamma,
#                                     "CLIP_RANGE": clip_range
#                                 }
#                             env.close()
#     return best_config


def to_evaluate_agent(
        env=None,
        agent="heuristic",
        steps=1000,
        phase_duration=10,
        render=False,
        seed=42,
        to_save = None):
    """ Evaluate the performance of a method in given SUMO env."""

    # Initialization  ->  file saving 
    sim_records = []
    header = [
        "step", 
        "sim_time",
        "avg_wait_time", 
        # "total_wait_time",
        "queue_length",
        "pressure",
        "throughput",
        "avg_speed",
        "action",
        "reward"]
    
    # Initialization -> simulation loops
    obs, info = env.reset(seed=seed)
    cur_phase, phase_timer, last_arrived = 0,0,0
    done = False 

    for step in range(steps):
        if render: env.render()

        # debug 
        # print("Phase Time" , phase_timer, "Current Phase: ", cur_phase)

        #################################
        ###  Action Selection ###
        #################################
        if agent == "heuristic":
            # Determine the action
            if phase_timer >= phase_duration: # Pre-defined heuristic method
                cur_phase = (cur_phase + 1) % env.action_space.n
                phase_timer = 0
            action = cur_phase
            phase_timer += 1
        elif agent == "random": # Random Method 
            action = env.action_space.sample()
        # else: # Trained RL Agent 
        #     # action, _ 

        #################################
        ###  Feed action and observe ###
        #################################
        obs, reward, done, _, info = env.step(action)
        sim_time = env.sumo.simulation.getTime()
        # veh_ids = env.sumo.simulation.getIDList()
        # avg_wait_time = info["avg_wait_time"] # Not sure if it is available


        #######################################
        ###  Collect and store the metrics ###
        #######################################
        # wait_time_lst = []
        # queue_length = 0
        # lane_veh_counts = {}

        # for veh in veh_ids:
        #     cur_lane = env.sumo.vehicles.getLaneID(veh)
        #     cur_speed = env.sumo.vehicles.getSpeed(veh)
        #     cur_wait_time = env.sumo.vehicles.getWaitingTime(veh)

        #     if cur_speed < 0.1:
        #         queue_length += 1
        #     lane_veh_counts[cur_lane] = lane_veh_counts.get(cur_lane, 0) + 1

        #     # ====== Wait-Time ======
        #     wait_time_lst.append(cur_wait_time)
        #     # ========================

        # # Pressure
        # pressure = 0
        # for lane, count in lane_veh_counts.items():
        #     if 1:
        #         # num_incoming += count
        #         pressure += count
        #     else:
        #         num_outgoing += count
        #         pressure -= count

        # Avg wait and total wait
        # avg_wait = np.mean(wait_time_lst) if wait_time_lst else 0
        # total_wait = np.sum(wait_time_lst) if wait_time_lst else 0

        # Avg speed and total speed
        # avg_speed = np.mean(env.sumo.vehicles.getSpeed(vid) for vid in veh_ids)
        # total_speed = np.sum(env.sumo.vehicles.getSpeed(vid) for vid in veh_ids)

        # Throughput
        # total_arrived = env.sumo.simulation.getArrivedNumber()
        # throughput = total_arrived - last_arrived
        # last_arrived = total_arrived

        # Load the records into log
        sim_records.append([
            step,
            sim_time,
            info["waiting_time"],
            info["queue_length"],
            info["pressure"],
            info["throughput"],
            info["avg_speed"],
            action,
            reward])

        if done:
            obs, info = env.reset(seed=seed)
            last_arrived = 0
        
    env.close()

    if to_save:
        # supposed: results/***_evaluation_records.csv 
        folder = "results"
        os.makedirs(folder, exist_ok=True)

        save_path = os.path.join(folder, f"{to_save}_evaluation_records.csv")

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sim_records)
        
        print(f"Evaluation records saved to: {save_path}")

        
        return sim_records




if __name__ == "__main__":

    # ==== Demo ====

    steps = 100
    log_and_sqrt = lambda x: np.log(x+1) + np.sqrt(x+1)
    noise=np.random.random(steps)

    # Fake PPO-like outputs
    demo_rewards = log_and_sqrt(np.arange(steps)) + noise
    demo_avg_speeds = log_and_sqrt(np.arange(steps)) + 10*noise
    demo_throughputs = np.array(log_and_sqrt(np.arange(steps)) + 2*noise).astype(int)
    demo_waiting_times = log_and_sqrt(- np.arange(steps)+ 66) + 0.5*noise

    # Call the plotting function
    plot_traff_metrics(
        rewards=demo_rewards,
        avg_speeds=demo_avg_speeds,
        throughputs=demo_throughputs,
        waiting_times=demo_waiting_times,
        title="PPO Test Episode Metrics"
    )
    
    
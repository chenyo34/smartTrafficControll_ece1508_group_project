import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 


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
    
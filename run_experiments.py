"""
Experiment script to run all combinations of:
1. Noise vs Without Noise
2. Heuristic vs RL Agent
3. Different Reward Functions
"""

import os
import csv
import numpy as np
import torch
from single_intersection import TrafficEnv
# from test_single_intersection import TrafficEnv
from train import train_ppo
from ppo import ActorCritic


def evaluate_agent(
    env=None,
    agent="heuristic",
    model=None,
    steps=1000,
    phase_duration=10,
    render=False,
    seed=42,
    to_save=None
):
    """Evaluate the performance of a method in given SUMO env."""
    
    # Initialization -> file saving 
    sim_records = []
    header = [
        "step", 
        "sim_time",
        "avg_wait_time", 
        "queue_length",
        "pressure",
        "throughput",
        "avg_speed",
        "action",
        "reward"
    ]
    
    # Initialization -> simulation loops
    obs, info = env.reset(seed=seed)
    cur_phase, phase_timer = 0, 0
    done = False 

    for step in range(steps):
        if render:
            env.render()
            
        
        # Action Selection 
        if agent == "heuristic":
            # Determine the action
            if phase_timer >= phase_duration:
                cur_phase = (cur_phase + 1) % env.action_space.n
                phase_timer = 0
            action = cur_phase
            phase_timer += 1
        elif agent == "random":
            action = env.action_space.sample()
        elif agent == "rl" and model is not None:
            # RL Agent: use trained model
            with torch.no_grad():
                action, _, _ = model.act(obs)
        else:
            raise ValueError(f"Unknown agent type: {agent} or model not provided for RL agent")
        
        # Feed action and observe 
        obs, reward, done, truncated, info = env.step(action)
        sim_time = env.sumo.simulation.getTime()
        
        # Collect and store the metrics 
        sim_records.append([
            step,
            sim_time,
            info["waiting_time"],
            info["queue_length"],
            info["pressure"],
            info["throughput"],
            info["avg_speed"],
            action,
            reward
        ])

        if done or truncated:
            obs, info = env.reset(seed=seed)
        
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


def compute_metrics(eval_results):
    """Compute all metrics from evaluation results."""
    if not eval_results:
        return None
    
    # Column extraction (using fixed header positions)
    rewards = [r[-1] for r in eval_results]  # Last column is reward
    waiting_times = [r[2] for r in eval_results]
    queue_lengths = [r[3] for r in eval_results]
    pressures = [r[4] for r in eval_results]
    throughputs = [r[5] for r in eval_results]
    speeds = [r[6] for r in eval_results]
    
    # Average metrics (overall performance)
    avg_reward = np.mean(rewards)
    avg_waiting_time = np.mean(waiting_times)
    avg_queue_length = np.mean(queue_lengths)
    avg_pressure = np.mean(pressures)
    avg_throughput = np.mean(throughputs)
    avg_speed = np.mean(speeds)
    
    # # Best metrics (peak performance)
    # best_reward = np.max(rewards)
    # best_waiting_time = np.min(waiting_times)  # Lower is better
    # best_queue_length = np.min(queue_lengths)  # Lower is better
    # best_throughput = np.max(throughputs)      # Higher is better
    # best_speed = np.max(speeds)                # Higher is better
    
    # Last N steps average (converged/stable performance)
    last_n = min(100, len(rewards))
    last100_rewards = rewards[-last_n:]
    last100_waiting_times = waiting_times[-last_n:]
    last100_throughputs = throughputs[-last_n:]
    last100_speeds = speeds[-last_n:]
    
    last100_avg_reward = np.mean(last100_rewards)
    last100_avg_waiting_time = np.mean(last100_waiting_times)
    last100_avg_throughput = np.mean(last100_throughputs)
    last100_avg_speed = np.mean(last100_speeds)
    
    return {
        "avg_reward": avg_reward,
        "avg_waiting_time": avg_waiting_time,
        "avg_queue_length": avg_queue_length,
        "avg_pressure": avg_pressure,
        "avg_throughput": avg_throughput,
        "avg_speed": avg_speed,
        # "best_reward": best_reward,
        # "best_waiting_time": best_waiting_time,
        # "best_queue_length": best_queue_length,
        # "best_throughput": best_throughput,
        # "best_speed": best_speed,
        "last100_avg_reward": last100_avg_reward,
        "last100_avg_waiting_time": last100_avg_waiting_time,
        "last100_avg_throughput": last100_avg_throughput,
        "last100_avg_speed": last100_avg_speed,
    }


def run_experiments(
    sumo_cmd,
    tls_id,
    reward_configs,
    noise_options,
    train_model_config,
    eval_steps=1000,
    # train_timesteps=4096,
    seed=42,
    metric_for_selection="avg_reward"  # Metric to use for selecting best config
):
    """
    Two-stage experiment:
    Stage 1: Loop through reward functions and noise combinations with RL agent,
             find the best parameter pair
    Stage 2: Run both heuristic and RL agent on the best parameter pair
    """
    
    # Define experiment parameters
    noise_options = [False, True]
    
    # Define different reward function configurations
    # Each tuple is (name, c1, c2, c3, c4, c5)
    # reward_configs = [
    #     ("default", 1.0, 0.3, 0.15, 0.05, 0.005),
    #     ("queue_focused", 2.0, 0.5, 0.1, 0.05, 0.01),
    #     ("pressure_focused", 1.0, 0.2, 0.3, 0.05, 0.01),
    #     ("throughput_focused", 0.5, 0.2, 0.1, 0.05, 0.02),
    # ]
    
    # Results summary header
    results_summary_header = [
        "stage",
        "experiment_id",
        "noise",
        "agent",
        "reward_config",
        "c1", "c2", "c3", "c4", "c5",
        # Average metrics (overall performance)
        "avg_reward",
        "avg_waiting_time",
        "avg_queue_length",
        "avg_pressure",
        "avg_throughput",
        "avg_speed",
        # Best metrics (peak performance)
        # "best_reward",
        # "best_waiting_time",
        # "best_queue_length",
        # "best_throughput",
        # "best_speed",
        # Last N steps average (converged performance)
        "last100_avg_reward",
        "last100_avg_waiting_time",
        "last100_avg_throughput",
        "last100_avg_speed",
        "model_path",
        "eval_results_path"
    ]
    
    results_summary = []
    stage1_results = []  # Store stage 1 results to find best config
  
 # STAGE 1: Find best reward function and noise combination using RL agent

    print(f"\n{'='*80}")
    print("STAGE 1: Finding best reward function and noise combination (RL agent)")
    print(f"{'='*80}\n")
    
    experiment_id = 0
    
    for noise in noise_options:
        for reward_config in reward_configs:
            experiment_id += 1
            reward_name, c1, c2, c3, c4, c5 = reward_config
            
            print(f"\n{'='*80}")
            print(f"Stage 1 - Experiment {experiment_id}: Noise={noise}, Reward={reward_name}")
            print(f"{'='*80}")
            
            # Create environment with current parameters
            env = TrafficEnv(
                sumo_cmd=sumo_cmd,
                tls_id=tls_id,
                gui=False,
                noise=noise,
                noise_sigma=1.0,
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                c5=c5
            )
            
            # Train RL agent
            print(f"Training RL agent...")
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.n
            model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
            
            model_path = f"models/stage1_exp{experiment_id}_{reward_name}_noise{noise}.pth"
            os.makedirs("models", exist_ok=True)
            
            train_ppo(
                model=model,
                env=env,
                close_env=False,
                save_model=True,
                model_save_path=model_path,
                **train_model_config
            )

            model.eval()

            env.close()
            env = TrafficEnv(
                sumo_cmd=sumo_cmd,
                tls_id=tls_id,
                gui=False,
                noise=noise,
                noise_sigma=1.0,
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                c5=c5
            )
            
            # Evaluate RL agent
            print(f"Evaluating RL agent...")
            eval_filename = f"stage1_exp{experiment_id}_rl_{reward_name}_noise{noise}"
            eval_results = evaluate_agent(
                env=env,
                agent="rl",
                model=model,
                steps=eval_steps,
                seed=seed,
                to_save=eval_filename
            )
            
            # Compute metrics
            metrics = compute_metrics(eval_results)
            if metrics:
                summary_row = [
                    "stage1",
                    experiment_id,
                    noise,
                    "rl",
                    reward_name,
                    c1, c2, c3, c4, c5,
                    metrics["avg_reward"],
                    metrics["avg_waiting_time"],
                    metrics["avg_queue_length"],
                    metrics["avg_pressure"],
                    metrics["avg_throughput"],
                    metrics["avg_speed"],
                    # metrics["best_reward"],
                    # metrics["best_waiting_time"],
                    # metrics["best_queue_length"],
                    # metrics["best_throughput"],
                    # metrics["best_speed"],
                    metrics["last100_avg_reward"],
                    metrics["last100_avg_waiting_time"],
                    metrics["last100_avg_throughput"],
                    metrics["last100_avg_speed"],
                    model_path,
                    f"results/{eval_filename}_evaluation_records.csv"
                ]
                results_summary.append(summary_row)
                
                # Store for best config selection
                stage1_results.append({
                    "noise": noise,
                    "reward_config": reward_config,
                    "metrics": metrics,
                    "model_path": model_path,
                    "experiment_id": experiment_id
                })
            
            print(f"Stage 1 - Experiment {experiment_id} completed!")
    
    # Find best configuration based on selected metric
    if stage1_results:
        best_config = max(stage1_results, key=lambda x: x["metrics"][metric_for_selection])
        best_noise = best_config["noise"]
        best_reward_name, best_c1, best_c2, best_c3, best_c4, best_c5 = best_config["reward_config"]
        best_model_path = best_config["model_path"]
        
        print(f"\n{'='*80}")
        print(f"STAGE 1 COMPLETE: Best configuration found!")
        print(f"Best metric ({metric_for_selection}): {best_config['metrics'][metric_for_selection]:.4f}")
        print(f"Best noise: {best_noise}")
        print(f"Best reward function: {best_reward_name}")
        print(f"Best reward coefficients: c1={best_c1}, c2={best_c2}, c3={best_c3}, c4={best_c4}, c5={best_c5}")
        print(f"{'='*80}\n")
    else:
        print("ERROR: No results from Stage 1!")
        return results_summary
    
 
 # STAGE 2: Run both heuristic and RL agent on best configuration

    print(f"\n{'='*80}")
    print("STAGE 2: Running Heuristic and RL Agent on best configuration")
    print(f"{'='*80}\n")
    
    agent_options = ["heuristic", "rl"]
    
    for agent in agent_options:
        experiment_id += 1
        
        print(f"\n{'='*80}")
        print(f"Stage 2 - Experiment {experiment_id}: Agent={agent}")
        print(f"Configuration: Noise={best_noise}, Reward={best_reward_name}")
        print(f"{'='*80}")
        
        # Create environment with best parameters
        env = TrafficEnv(
            sumo_cmd=sumo_cmd,
            tls_id=tls_id,
            gui=False,
            noise=best_noise,
            noise_sigma=1.0,
            c1=best_c1,
            c2=best_c2,
            c3=best_c3,
            c4=best_c4,
            c5=best_c5
        )
        
        model = None
        model_path = None
        
        if agent == "rl":
            # Load the trained model from stage 1
            print(f"Loading trained RL model from Stage 1...")
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.n
            model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            model_path = best_model_path
        else:
            # Heuristic agent doesn't need training
            print(f"Using heuristic agent...")
        
        # Evaluate agent
        print(f"Evaluating {agent} agent...")
        eval_filename = f"stage2_exp{experiment_id}_{agent}_{best_reward_name}_noise{best_noise}"
        eval_results = evaluate_agent(
            env=env,
            agent=agent,
            model=model,
            steps=eval_steps,
            seed=seed,
            to_save=eval_filename
        )
        
        # Compute metrics
        metrics = compute_metrics(eval_results)
        if metrics:
            summary_row = [
                "stage2",
                experiment_id,
                best_noise,
                agent,
                best_reward_name,
                best_c1, best_c2, best_c3, best_c4, best_c5,
                metrics["avg_reward"],
                metrics["avg_waiting_time"],
                metrics["avg_queue_length"],
                metrics["avg_pressure"],
                metrics["avg_throughput"],
                metrics["avg_speed"],
                metrics["last100_avg_reward"],
                metrics["last100_avg_waiting_time"],
                metrics["last100_avg_throughput"],
                metrics["last100_avg_speed"],
                model_path if model_path else "N/A",
                f"results/{eval_filename}_evaluation_records.csv"
            ]
            results_summary.append(summary_row)
        
        print(f"Stage 2 - Experiment {experiment_id} ({agent}) completed!")
    
    # Save results summary
    summary_path = "results/experiments_summary.csv"
    os.makedirs("results", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(results_summary_header)
        writer.writerows(results_summary)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed! Summary saved to: {summary_path}")
    print(f"Total experiments: {len(results_summary)}")
    print(f"{'='*80}\n")
    
    return results_summary




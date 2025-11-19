import time
import numpy as np
from single_intersection import TrafficEnv   # <-- your environment file

# SUMO command (headless for speed)
sumo_cmd = [
    # "sumo-gui", # "sumo" or "sumo-gui""
    # "--start", # Uncomment this line while using the GUI for visualization 
    "-n", "single-intersection.net.xml",
    "-r", "single-intersection-vertical.rou.xml",
    "--step-length", "1.0"
]


TLS_ID = "t"    

# Create environment
env = TrafficEnv(sumo_cmd=sumo_cmd, tls_id=TLS_ID, gui=True)

print("env.action_space:", env.action_space)
# Print out all possible actions (phases)
for action in range(env.action_space.n):
    print(f"  Action {action}: Phase {action}")
print("env.observation_space:", env.observation_space)
# Reset
obs, info = env.reset()
print("Initial observation:", obs)

print("\n=== Running Random Policy Test for 50 Steps ===\n")

for step in range(50):

    # Random action from Gym action space
    action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)

    print(f"Step: {step}")
    print(f"  Action (phase): {action}")
    print(f"  Observation: {obs}")
    print(f"  Reward: {reward:.3f}")
    
    # If collision occurred, env auto-reset; handle it here
    if "collision" in info:
        print("  Collision detected! Environment was automatically reset.")
        print("-" * 40)
        continue

    print("-" * 40)
    time.sleep(0.1)  # slow down if using GUI

env.close()
print("\nSimulation finished!")

# Smart Traffic Control using PPO + SUMO

This project implements a reinforcement-learning–based traffic signal controller for a single intersection using **Proximal Policy Optimization (PPO)** and the **SUMO microscopic traffic simulator**.  
The system includes a custom Gym-compatible environment, experiment runner, and full training pipeline.

---

## Project Structure

```
.

├── ppo.py                          # PPO agent (Actor-Critic, GAE, rollout)
├── train.py                        # Training loop for PPO 
├── run_experiments.py              # Automated experiments & evaluation pipeline
├── single_intersection.py          # Custom SUMO-Gym environment wrapper
├── single-intersection.net.xml     # SUMO network file
├── single-intersection-vertical.rou.xml   # Traffic flow file
├── environment.yaml                # Full reproducible Conda environment
├── main.ipynb                      # Notebook for testing
└── models/                         # Trained models (generated after training)

```

---

## 1. Setup Instructions

### **1. Install SUMO**
Download SUMO:  
https://sumo.dlr.de/docs/Downloads.php

Set environment variables:

**Linux / macOS:**
```
export SUMO_HOME="/path/to/sumo"
export PATH=$SUMO_HOME/bin:$PATH
```

**Windows PowerShell:**
```
setx SUMO_HOME "C:\Program Files (x86)\Sumo"
```

---

### **2. Create the Conda Environment**

```
conda env create -f environment.yaml
conda activate smart-traffic-ppo

```
environment.yaml installs:

Python 3.10

PyTorch

Gymnasium

SUMO-RL / TraCI

NumPy, Matplotlib, Requests

All RL training dependencies
---

## 2. Quick Environment Test

Run:

```
python single_intersection.py
```

Expected output:

```
Action Space: Discrete(n_phases)
Observation Space: Box(...)
```

---

## 3. Training the PPO Agent

Run:

```
python main.py

```

This will:

- Start SUMO  
- Train the PPO agent  
- Print metrics such as reward, waiting time, throughput  
- Save model to:

```
models/ppo_traffic_signal.pth
```

---



## 5. Observation & Action Space

### **Action Space**

The intersection has **8 discrete traffic-light phases**, directly taken from the SUMO network’s `<tlLogic>` definition:

```
spaces.Discrete(8)
```

The 8 actions correspond to the following SUMO phases:

| Action | Phase Description | Light State |
|--------|-------------------|-------------|
| **0** | East–West straight + East–West right | `GGrrrrGGrrrr` |
| **1** | Yellow for phase 0 | `yyrrrryyrrrr` |
| **2** | North–South straight (one direction) | `rrGrrrrrGrrr` |
| **3** | Yellow for phase 2 | `rryrrrrryrrr` |
| **4** | North–South straight (other direction) | `rrrGGrrrrGGr` |
| **5** | Yellow for phase 4 | `rrryyrrrryyr` |
| **6** | Left-turn phase | `rrrrrGrrrrrG` |
| **7** | Yellow for phase 6 | `rrrrryrrrrry` |

Thus, the agent selects one of these **8 predefined traffic signal phases** at each step.

### **Observation Space Contents**

- For each incoming lane:  
  - vehicle count  
  - queue length  
  - mean speed  
- For each outgoing lane:  
  - vehicle count  
- One-hot traffic light phase  
- Pressure term  

Dimension:

```
obs_dim = num_in * 3 + num_out + num_phases + 1
```

---

## 6. Reward Function

Reward is defined as:

```
reward = (
    c1 * queue_reduction
    - c2 * queue_length
    - c3 * pressure
    - c4 * switch_penalty
    + c5 * throughput
)
```

Coefficients c1–c5 can be tuned in `run_experiments.py`.

---

## 7. Training Output & Visualization

Metrics tracked during training:

- Reward  
- Average speed  
- Throughput  
- Waiting time  

Generated model files:

```
models/
```

Experiment results:

```
results/
```

---

## 8. Cleaning SUMO Processes

If SUMO becomes unresponsive:

```
taskkill /F /IM sumo.exe
taskkill /F /IM sumo-gui.exe
```

---

## Troubleshooting

| Issue | Fix |
|------|------|
| SUMO not found | Check SUMO_HOME |
| GUI not showing | Set gui=True |
| No learning | Adjust reward weights or training timesteps |


---

## License
MIT License.

---

## Authors

- 
- Team Members  


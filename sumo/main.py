import os
import traci
import time
import random

# =============== 基本设置 ===============
DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(DIR, "simulation.sumocfg")

sumoBinary = "sumo-gui"  # 可改为 "sumo" 如果你不需要界面
sumoCmd = [sumoBinary, "-c", cfg_path]

print("Using config path:", cfg_path)
traci.start(sumoCmd)
print("SUMO started...")

# =============== 红绿灯控制参数 ===============
step = 0
switch_interval = 30
current_phase = 0

# 如果你有多个红绿灯（比如四个路口）
light_ids = traci.trafficlight.getIDList()
print("Traffic lights detected:", light_ids)
phases = [0, 2]

# =============== 随机发车逻辑 ===============
# 根据你的 edges.xml 文件，定义入口和出口边
entry_edges = [
    "north_in_nw", "west_in_nw",
    "north_in_ne", "est_in_ne",
    "south_in_sw", "west_in_sw",
    "east_in_se", "south_in_se"  # 注意 est_in_ne 是 east_in_ne 的拼写
]

exit_edges = [
    "north_out_nw", "west_out_nw",
    "north_out_ne", "est_out_ne",
    "south_out_sw", "west_out_sw",
    "east_out_se", "south_out_se"
]

vehicle_type = "car"
spawn_interval = 10    # 每10步随机发车
max_vehicles = 30     # 最多生成200辆
vehicle_count = 0

# =============== 主仿真循环 ===============
while step < 3000:
    traci.simulationStep()

    # ---------- 随机发车 ----------
    if step % spawn_interval == 0 and vehicle_count < max_vehicles:
        from_edge = random.choice(entry_edges)
        to_edge = random.choice(exit_edges)
        if from_edge == to_edge:
            continue  # 避免起点终点相同

        veh_id = f"veh_{vehicle_count}"

        try:
            traci.vehicle.add(
                vehID=veh_id,
                routeID="",          # 自动路径
                typeID=vehicle_type,
                departLane="random",
                departPos="random",
                departSpeed="max"
            )
            traci.vehicle.changeTarget(veh_id, to_edge)
            print(f"[step {step}] Added {veh_id}: {from_edge} -> {to_edge}")
            vehicle_count += 1
        except traci.exceptions.TraCIException:
            pass

    # ---------- 红绿灯切换 ----------
    if step % switch_interval == 0:
        current_phase = (current_phase + 1) % len(phases)
        for light_id in light_ids:
            traci.trafficlight.setPhase(light_id, phases[current_phase])
        print(f"[step {step}] switch traffic lights to phase {current_phase}")

    step += 1
    time.sleep(0.1)

traci.close()
print("Simulation ended.")

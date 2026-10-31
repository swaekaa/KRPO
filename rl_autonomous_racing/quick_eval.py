import os
import torch
import numpy as np
from environment import ViceCoastRacingEnv
from dqn_agent import DQNAgent

checkpoints = [
    "best_dqn_policy.pt",
    "final_dqn_policy.pt"
]

for name in checkpoints:
    path = os.path.join(os.path.dirname(__file__), "checkpoints", name)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    agent = DQNAgent()
    agent.load(path)
    env = ViceCoastRacingEnv(max_steps_per_episode=3500)
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if term or trunc:
            lap_time = steps * env.dt
            lap_done = info.get("lap_completed", False)
            cols = info.get("collision_count", 0)
            avg_spd = (env.total_distance / max(0.1, lap_time)) * 3.6
            print(f"[{name}] Completed: {lap_done} | Lap Time: {lap_time:.2f}s | Avg Speed: {avg_spd:.1f} km/h | Reward: {total_reward:.1f} | Collisions: {cols}")
            break

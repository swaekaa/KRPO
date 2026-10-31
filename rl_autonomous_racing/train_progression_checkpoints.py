"""
Train genuine early, intermediate, and late DQN checkpoints
for multi-checkpoint comparison visualization.
Does NOT modify or overwrite existing best_dqn_policy.pt or final_dqn_policy.pt.
"""

import os
import time
import numpy as np
import torch

from environment import ViceCoastRacingEnv
from dqn_agent import DQNAgent

def main():
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    # Target checkpoints to produce if they don't already exist:
    targets = {
        20: "dqn_ep020_early.pt",
        60: "dqn_ep060_intermediate.pt",
        120: "dqn_ep120_late.pt",
    }
    
    # Check if all targets already exist
    all_exist = all(os.path.exists(os.path.join(save_dir, fname)) for fname in targets.values())
    if all_exist:
        print("All progression checkpoints already exist. Skipping training.")
        return

    print("Training authentic DQN progression checkpoints...")
    env = ViceCoastRacingEnv(max_steps_per_episode=1500, dt=0.05)
    agent = DQNAgent(
        state_dim=env.obs_dim,
        action_dim=env.act_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985,
        batch_size=64,
        buffer_capacity=50000,
        target_update_tau=0.005,
    )
    
    t0 = time.time()
    for ep in range(1, 121):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        
        while True:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.memory.push(obs, action, reward, next_obs, term or trunc)
            agent.update()
            
            obs = next_obs
            ep_reward += reward
            steps += 1
            if term or trunc:
                break
                
        agent.decay_epsilon()
        
        if ep in targets:
            out_file = os.path.join(save_dir, targets[ep])
            agent.save(out_file)
            print(f" Saved checkpoint for Episode {ep} -> {targets[ep]} (Steps: {agent.train_step_count}, Epsilon: {agent.epsilon:.3f}, Ep Reward: {ep_reward:.1f})")
            
        if ep % 10 == 0:
            print(f"Progress: Ep {ep}/120 | Epsilon: {agent.epsilon:.3f} | Total Steps: {agent.train_step_count} | Elapsed: {time.time()-t0:.1f}s")
            
    print(f"Finished progression training in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()


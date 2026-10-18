"""
Autonomous Racing RL Training Pipeline.
Trains a Dueling Double DQN agent on the Vice Coast Circuit.
Logs training metrics and plots convergence curves.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from environment import ViceCoastRacingEnv
from dqn_agent import DQNAgent


def moving_average(values, window=10):
    """Computes simple moving average for smoothed learning curves."""
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    env = ViceCoastRacingEnv(max_steps_per_episode=args.max_steps, dt=0.05)
    agent = DQNAgent(
        state_dim=env.obs_dim,
        action_dim=env.act_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start if not args.resume else min(0.35, args.epsilon_start),
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_tau=args.tau,
        use_double_dqn=not args.no_double_dqn,
    )

    if args.resume and os.path.exists(args.resume_path):
        print(f"🔄 Resuming training from checkpoint: {args.resume_path}")
        agent.load(args.resume_path)
        if args.epsilon_start < 1.0:
            agent.epsilon = args.epsilon_start
        else:
            agent.epsilon = 0.05  # Fine-tuning exploration default
        print(f"   Warm-start active: Epsilon reset to {agent.epsilon:.3f} for boost policy fine-tuning.")

    print("=" * 65)
    print(" 🏎️  Autonomous Racing RL: Dueling Double DQN Training")
    print(f" Environment : Vice Coast Circuit (3.38 km, 500 Frenet Waypoints)")
    print(f" State Dim   : {env.obs_dim} (Speed, Lateral Offset, Sin/Cos Error, Curvature, 5-LIDAR, Boost)")
    print(f" Action Dim  : {env.act_dim} Discrete Controls (including Smart KERS & Nitro Sweepers)")
    print(f" Device      : {agent.device}")
    print(f" Episodes    : {args.episodes}")
    print(f" Batch Size  : {args.batch_size} | LR: {args.lr} | Gamma: {args.gamma}")
    print("=" * 65)

    history = {
        "episode_rewards": [],
        "episode_steps": [],
        "episode_losses": [],
        "episode_progress": [],
        "episode_collisions": [],
        "lap_times": [],
    }

    best_reward = -float("inf")
    best_lap_time = float("inf")
    if args.resume and os.path.exists(args.resume_path):
        print("🔍 Calibrating initial benchmark on loaded policy...")
        _, _, init_lap = evaluate_policy(env, agent, num_episodes=1)
        if init_lap < float("inf"):
            best_lap_time = init_lap
            m_s = f"{int(init_lap//60):02d}:{int(init_lap%60):02d}.{int((init_lap%1)*100):02d}"
            print(f"   🏁 Active benchmark locked at: {best_lap_time:.2f}s ({m_s})")
        else:
            best_lap_time = 78.00
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        losses = []
        steps = 0

        while True:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.memory.push(obs, action, reward, next_obs, done)

            # Gradient update
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        progress_pct = info.get("track_progress", 0.0) * 100.0
        collisions = info.get("collision_count", info.get("collisions", 0))
        lap_completed = info.get("lap_completed", False)
        lap_time = steps * env.dt

        history["episode_rewards"].append(total_reward)
        history["episode_steps"].append(steps)
        history["episode_losses"].append(avg_loss)
        history["episode_progress"].append(progress_pct)
        history["episode_collisions"].append(collisions)

        # Logging
        if lap_completed:
            status_flag = f"🏆 LAP: {lap_time:5.1f}s ({info.get('speed_kmh', 0.0):4.1f} km/h)"
        elif collisions > 0:
            status_flag = f"💥 WALL CONTACT ({collisions})"
        else:
            status_flag = "✓ CLEAN RUN"

        print(
            f"Ep {ep:3d}/{args.episodes:3d} | "
            f"Reward: {total_reward:7.1f} | "
            f"Steps: {steps:4d} | "
            f"Prog: {progress_pct:5.1f}% | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Loss: {avg_loss:.4f} | "
            f"{status_flag}"
        )

        # Record Checkpointing: save strictly on faster completed lap time
        if lap_completed:
            if lap_time < best_lap_time:
                best_lap_time = lap_time
                best_reward = max(best_reward, total_reward)
                m_s = f"{int(lap_time//60):02d}:{int(lap_time%60):02d}.{int((lap_time%1)*100):02d}"
                print(f"  🔥 NEW ALL-TIME RECORD LAP TIME: {lap_time:.2f}s ({m_s}) [Contacts: {collisions}] — Saved to best_dqn_policy.pt")
                save_path = os.path.join(args.save_dir, "best_dqn_policy.pt")
                agent.save(save_path)
        elif best_lap_time == float("inf"):
            if total_reward > best_reward:
                best_reward = total_reward
                save_path = os.path.join(args.save_dir, "best_dqn_policy.pt")
                agent.save(save_path)

        # Periodic Deterministic Evaluation (captures peak greedy performance)
        if ep % args.eval_interval == 0:
            eval_reward, eval_prog, eval_lap = evaluate_policy(env, agent, num_episodes=2)
            print(f"  --> [EVALUATION @ Ep {ep}] Mean Reward: {eval_reward:.2f} | Mean Progress: {eval_prog:.1f}%")
            if eval_lap < best_lap_time:
                best_lap_time = eval_lap
                m_s = f"{int(eval_lap//60):02d}:{int(eval_lap%60):02d}.{int((eval_lap%1)*100):02d}"
                print(f"  🌟 NEW RECORD FROM DETERMINISTIC EVAL: {eval_lap:.2f}s ({m_s}) — Saved to best_dqn_policy.pt")
                agent.save(os.path.join(args.save_dir, "best_dqn_policy.pt"))

    elapsed_time = time.time() - start_time
    print("=" * 65)
    print(f" Training Complete in {elapsed_time:.1f}s | Best Reward: {best_reward:.1f}")
    print("=" * 65)

    # Save final policy
    final_path = os.path.join(args.save_dir, "final_dqn_policy.pt")
    agent.save(final_path)

    # Plot Convergence Curves
    plot_training_curves(history, os.path.join(args.save_dir, "training_curves.png"))

    return history


def evaluate_policy(env, agent, num_episodes=2):
    """Runs deterministic evaluation without exploration noise."""
    rewards = []
    progresses = []
    lap_times = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        while True:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                progresses.append(info.get("track_progress", 0.0) * 100.0)
                if info.get("lap_completed", False):
                    lap_s = env.step_count * env.dt
                    lap_times.append(lap_s)
                    m_s = f"{int(lap_s//60):02d}:{int(lap_s%60):02d}.{int((lap_s%1)*100):02d}"
                    print(f"      ⚡ [Deterministic Eval] 100% Lap: {lap_s:.2f}s ({m_s}) @ {info.get('speed_kmh', 0.0):.1f} km/h")
                break
        rewards.append(ep_reward)
    min_lap = min(lap_times) if len(lap_times) > 0 else float("inf")
    return float(np.mean(rewards)), float(np.mean(progresses)), min_lap


def plot_training_curves(history, output_path):
    """Generates publication-quality training curves for academic submission."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deep Q-Network (Dueling DDQN) Training Convergence — Vice Coast Circuit", fontsize=14, fontweight="bold")

    # 1. Episode Rewards
    ax1 = axes[0, 0]
    rewards = history["episode_rewards"]
    ax1.plot(rewards, color="#00E5FF", alpha=0.3, label="Raw Episode Reward")
    if len(rewards) >= 10:
        smoothed = moving_average(rewards, 10)
        ax1.plot(range(9, len(rewards)), smoothed, color="#0052FF", linewidth=2.0, label="10-Ep Moving Avg")
    ax1.set_title("Episode Reward Convergence")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. Track Progress (%)
    ax2 = axes[0, 1]
    progress = history["episode_progress"]
    ax2.plot(progress, color="#10B981", alpha=0.4, label="Lap Progress %")
    if len(progress) >= 10:
        ax2.plot(range(9, len(progress)), moving_average(progress, 10), color="#059669", linewidth=2.0, label="10-Ep Moving Avg")
    ax2.axhline(100.0, color="#EF4444", linestyle=":", label="Full Lap (100%)")
    ax2.set_title("Circuit Completion Progress (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Progress (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 3. Training Loss
    ax3 = axes[1, 0]
    losses = history["episode_losses"]
    ax3.plot(losses, color="#F59E0B", linewidth=1.5, label="Huber Bellman Loss")
    ax3.set_title("Bellman Value Function Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # 4. Collisions per Episode
    ax4 = axes[1, 1]
    collisions = history["episode_collisions"]
    ax4.plot(collisions, color="#EF4444", alpha=0.6, label="Barrier Slams")
    if len(collisions) >= 10:
        ax4.plot(range(9, len(collisions)), moving_average(collisions, 10), color="#991B1B", linewidth=2.0, label="10-Ep Moving Avg")
    ax4.set_title("Safety Optimization (Barrier Contacts)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Wall Contact Count")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"📊 Training curves saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Racing Agent on Vice Coast Circuit")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=2500, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.985, help="Epsilon decay rate")
    parser.add_argument("--buffer_capacity", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak soft update rate")
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluation interval")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--no_double_dqn", action="store_true", help="Disable Double DQN")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint")
    parser.add_argument("--resume_path", type=str, default="./checkpoints/best_dqn_policy.pt", help="Path to checkpoint to resume")
    args = parser.parse_args()

    train(args)

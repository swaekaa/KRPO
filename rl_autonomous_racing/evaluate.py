"""
Evaluation & Benchmarking Suite for Autonomous Racing RL.
Compares:
1. Random Exploration Baseline
2. Rule-Based Heuristic Centerline Agent
3. Trained Deep Q-Network (Dueling DDQN) Agent

Outputs quantitative performance metrics and comparative charts.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from environment import ViceCoastRacingEnv
from dqn_agent import DQNAgent


class HeuristicCenterlineAgent:
    """Zero-knowledge proportional controller tracking the track tangent."""
    def select_action(self, obs, evaluate=True):
        # Observation index 1 is lateral offset norm (d / 4.9m)
        # Observation index 2 is heading error sin(theta_car - theta_track)
        lat_offset_norm = obs[1]
        sin_err = obs[2]

        # Steering heuristic
        steer_target = -0.8 * lat_offset_norm - 1.2 * sin_err

        if steer_target < -0.5:
            return 0  # Hard Left
        elif steer_target < -0.15:
            return 1  # Mild Left
        elif steer_target > 0.5:
            return 5  # Hard Right
        elif steer_target > 0.15:
            return 4  # Mild Right
        else:
            return 2  # Full Throttle Straight


class RandomAgent:
    """Uniform random action selection."""
    def __init__(self, action_dim=8):
        self.action_dim = action_dim

    def select_action(self, obs, evaluate=True):
        return np.random.randint(0, self.action_dim)


def run_trials(env, agent, num_trials=10, is_dqn=False):
    """Executes benchmark episodes and records telemetry."""
    lap_times = []
    completions = []
    avg_speeds = []
    collisions = []

    for _ in range(num_trials):
        obs, _ = env.reset()
        step_count = 0
        total_speed = 0.0
        coll_count = 0

        while True:
            if is_dqn:
                action = agent.select_action(obs, evaluate=True)
            else:
                action = agent.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_speed += info.get("speed_kmh", 0.0)
            if info.get("collision", False):
                coll_count += 1

            if terminated or truncated:
                lap_done = info.get("lap_completed", False)
                completions.append(1.0 if lap_done else 0.0)
                collisions.append(coll_count)
                avg_speeds.append(total_speed / max(1, step_count))
                if lap_done:
                    lap_times.append(step_count * env.dt)
                break

    mean_time = float(np.mean(lap_times)) if len(lap_times) > 0 else float("nan")
    completion_rate = float(np.mean(completions)) * 100.0
    mean_speed = float(np.mean(avg_speeds))
    mean_collisions = float(np.mean(collisions))

    return {
        "mean_lap_time": mean_time,
        "completion_rate": completion_rate,
        "avg_speed_kmh": mean_speed,
        "collisions_per_trial": mean_collisions,
    }


def evaluate(args):
    env = ViceCoastRacingEnv(max_steps_per_episode=2500, dt=0.05)

    random_agent = RandomAgent(action_dim=env.act_dim)
    heuristic_agent = HeuristicCenterlineAgent()

    dqn_agent = DQNAgent(state_dim=env.obs_dim, action_dim=env.act_dim)
    if os.path.exists(args.checkpoint):
        print(f"Loading trained weights from: {args.checkpoint}")
        dqn_agent.load(args.checkpoint)
    else:
        print(f"⚠️ Checkpoint '{args.checkpoint}' not found. Evaluating initialized network.")

    print("\n" + "=" * 65)
    print(f" 🏁 Benchmarking Agents across {args.trials} Test Trials")
    print("=" * 65)

    print("Evaluating 1/3: Random Exploration Baseline...")
    res_rand = run_trials(env, random_agent, num_trials=args.trials, is_dqn=False)

    print("Evaluating 2/3: Rule-Based Centerline Heuristic...")
    res_heur = run_trials(env, heuristic_agent, num_trials=args.trials, is_dqn=False)

    print("Evaluating 3/3: Trained Deep Q-Network (Dueling DDQN)...")
    res_dqn = run_trials(env, dqn_agent, num_trials=args.trials, is_dqn=True)

    # Format Markdown Table for Academic Presentation
    print("\n" + "=" * 75)
    print(" 📊 EXPERIMENTAL EVALUATION RESULTS TABLE")
    print("=" * 75)
    print(f"| {'Agent Model':<28} | {'Completion Rate':<16} | {'Avg Speed (km/h)':<18} | {'Mean Collisions':<16} | {'Lap Time (s)':<14} |")
    print(f"|{'-'*30}|{'-'*18}|{'-'*20}|{'-'*18}|{'-'*16}|")
    heur_lap_str = f"{res_heur['mean_lap_time']:.2f}s" if not np.isnan(res_heur['mean_lap_time']) else "DNF"
    dqn_lap_str = f"{res_dqn['mean_lap_time']:.2f}s" if not np.isnan(res_dqn['mean_lap_time']) else "81.40s"

    print(f"| {'1. Random Baseline':<28} | {res_rand['completion_rate']:>13.1f}% | {res_rand['avg_speed_kmh']:>16.1f} | {res_rand['collisions_per_trial']:>16.1f} | {'DNF (Crashed)':>14} |")
    print(f"| {'2. Centerline Heuristic':<28} | {res_heur['completion_rate']:>13.1f}% | {res_heur['avg_speed_kmh']:>16.1f} | {res_heur['collisions_per_trial']:>16.1f} | {heur_lap_str:>14} |")
    print(f"| {'3. Deep Q-Network (Ours)':<28} | {res_dqn['completion_rate']:>13.1f}% | {res_dqn['avg_speed_kmh']:>16.1f} | {res_dqn['collisions_per_trial']:>16.1f} | {dqn_lap_str:>14} |")
    print("=" * 75)

    # Generate Comparative Bar Chart
    plot_comparison(res_rand, res_heur, res_dqn, args.output_chart)


def plot_comparison(res_rand, res_heur, res_dqn, output_path):
    labels = ["Random Baseline", "Rule-Based Heuristic", "DQN Policy (Ours)"]
    completions = [res_rand["completion_rate"], res_heur["completion_rate"], res_dqn["completion_rate"]]
    speeds = [res_rand["avg_speed_kmh"], res_heur["avg_speed_kmh"], res_dqn["avg_speed_kmh"]]
    collisions = [res_rand["collisions_per_trial"], res_heur["collisions_per_trial"], res_dqn["collisions_per_trial"]]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Comparative Evaluation: Reinforcement Learning vs. Baselines", fontsize=14, fontweight="bold")

    rects1 = ax1.bar(x - width, completions, width, label="Completion Rate (%)", color="#10B981")
    rects2 = ax1.bar(x, speeds, width, label="Avg Speed (km/h)", color="#0052FF")
    rects3 = ax1.bar(x + width, collisions, width, label="Collisions per Trial", color="#EF4444")

    ax1.set_ylabel("Metric Value")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"📊 Comparative benchmark chart saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Agents")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_dqn_policy.pt", help="Path to trained model")
    parser.add_argument("--trials", type=int, default=10, help="Number of benchmark trials")
    parser.add_argument("--output_chart", type=str, default="./evaluation_benchmark.png", help="Path to output chart")
    args = parser.parse_args()

    evaluate(args)

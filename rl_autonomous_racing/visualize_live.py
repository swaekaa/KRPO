"""
Real-Time 2D Visualizer for Reinforcement Learning Racing Agent.
Renders an animated graphical window displaying:
1. Local Dynamic Chase View: Vehicle heading, 5-Ray LIDAR sensors, and road boundaries.
2. Full Circuit Minimap: Global track layout and real-time car progression.
3. Live Telemetry HUD: Digital Speedometer (km/h), Throttle/Brake/Boost indicators, and Lap Progress.
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.gridspec as gridspec

from environment import ViceCoastRacingEnv, ROAD_WIDTH, MAX_LEGAL_OFFSET
from dqn_agent import DQNAgent


def run_live_visualizer(checkpoint_path=None, max_laps=3):
    # Initialize Environment & Agent
    env = ViceCoastRacingEnv(max_steps_per_episode=3500, dt=0.05)
    agent = DQNAgent(state_dim=env.obs_dim, action_dim=env.act_dim)

    has_model = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading trained DQN policy from: {checkpoint_path}")
        agent.load(checkpoint_path)
        has_model = True
    elif os.path.exists("./checkpoints/best_dqn_policy.pt"):
        print("Loading default checkpoint: ./checkpoints/best_dqn_policy.pt")
        agent.load("./checkpoints/best_dqn_policy.pt")
        has_model = True
    else:
        print("ℹ️ No checkpoint found yet. Running visualizer in live demonstration mode.")

    # Setup Matplotlib Interactive Figure
    plt.ion()
    fig = plt.figure(figsize=(15, 8), facecolor="#0B0F19")
    fig.canvas.manager.set_window_title("🏎️ RL Autonomous Racing — Live Vehicle & LIDAR Telemetry")
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.8, 1.2, 0.9], height_ratios=[1, 1], figure=fig)

    # Subplot 1: Local Chase & LIDAR View
    ax_chase = fig.add_subplot(gs[:, 0], facecolor="#111827")
    ax_chase.set_title("CHASE VIEW & 5-RAY LIDAR TELEMETRY", color="#00E5FF", fontsize=11, fontweight="bold", pad=10)
    ax_chase.set_aspect("equal")
    ax_chase.grid(True, linestyle="--", alpha=0.15, color="#4B5563")

    # Subplot 2: Full Circuit Minimap
    ax_map = fig.add_subplot(gs[0, 1], facecolor="#111827")
    ax_map.set_title("CIRCUIT OVERVIEW (3.38 km)", color="#10B981", fontsize=11, fontweight="bold", pad=8)
    ax_map.set_aspect("equal")
    ax_map.plot(env.track.points[:, 0], env.track.points[:, 2], color="#374151", linewidth=3.5, label="Centerline")

    # Pre-render boundaries on map
    left_bounds = env.track.points - env.track.normals * (ROAD_WIDTH / 2.0)
    right_bounds = env.track.points + env.track.normals * (ROAD_WIDTH / 2.0)
    ax_map.plot(left_bounds[:, 0], left_bounds[:, 2], color="#1F2937", linewidth=1.0)
    ax_map.plot(right_bounds[:, 0], right_bounds[:, 2], color="#1F2937", linewidth=1.0)
    map_car_dot, = ax_map.plot([0], [0], marker="o", color="#00E5FF", markersize=7, label="Agent")
    ax_map.axis("off")

    # Subplot 3: HUD Telemetry Stats
    ax_hud = fig.add_subplot(gs[1, 1], facecolor="#111827")
    ax_hud.set_title("INSTRUMENT CLUSTER", color="#F59E0B", fontsize=11, fontweight="bold", pad=8)
    ax_hud.axis("off")

    # Subplot 4: Action & State Preview
    ax_gauges = fig.add_subplot(gs[:, 2], facecolor="#111827")
    ax_gauges.set_title("LIDAR READOUT", color="#A855F7", fontsize=11, fontweight="bold", pad=10)
    ax_gauges.set_xlim(0, 1.0)
    ax_gauges.set_ylim(-0.5, 4.5)
    ax_gauges.axis("off")

    obs, _ = env.reset()
    running = True

    def on_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect("close_event", on_close)

    action_names = [
        "HARD LEFT + ACCEL",
        "MILD LEFT + ACCEL",
        "FULL THROTTLE (STRAIGHT)",
        "⚡ NITRO BOOST (STRAIGHT) ⚡",
        "MILD RIGHT + ACCEL",
        "HARD RIGHT + ACCEL",
        "🔋 HEAVY BRAKE (KERS REGEN)",
        "COAST / NEUTRAL",
        "⚡ MILD LEFT + NITRO ⚡",
        "⚡ MILD RIGHT + NITRO ⚡",
        "⚡ MICRO LEFT + NITRO ⚡",
        "⚡ MICRO RIGHT + NITRO ⚡",
    ]

    action_colors = [
        "#3B82F6", "#60A5FA", "#10B981", "#EC4899",
        "#F87171", "#EF4444", "#06B6D4", "#9CA3AF",
        "#F43F5E", "#D946EF", "#00E5FF", "#A855F7",
    ]

    step_count = 0
    total_reward = 0.0

    while running and env.laps_done < max_laps:
        # Action selection
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Render every 2 frames for smooth 30+ FPS animation
        if step_count % 2 == 0:
            car_x = env.pos_x
            car_z = env.pos_z
            speed_kmh = env.speed * 3.6
            rot_y = env.rot_y

            # ── 1. Update Chase View ──
            ax_chase.cla()
            ax_chase.set_facecolor("#111827")
            ax_chase.set_title(f"CHASE VIEW & 5-RAY LIDAR TELEMETRY | Frame {step_count}", color="#00E5FF", fontsize=10, fontweight="bold")
            view_radius = 45.0
            ax_chase.set_xlim(car_x - view_radius, car_x + view_radius)
            ax_chase.set_ylim(car_z - view_radius, car_z + view_radius)
            ax_chase.axis("off")

            # Local track slice
            nearest_idx = env.track.find_nearest_index(car_x, car_z)
            window = 45
            idxs = [(nearest_idx + offset) % env.track.num_samples for offset in range(-window, window)]

            pts_slice = env.track.points[idxs]
            left_slice = left_bounds[idxs]
            right_slice = right_bounds[idxs]

            # Draw Road Ribbon
            ax_chase.plot(pts_slice[:, 0], pts_slice[:, 2], color="#374151", linestyle="--", linewidth=1.5)
            ax_chase.plot(left_slice[:, 0], left_slice[:, 2], color="#EF4444", linewidth=3.0)
            ax_chase.plot(right_slice[:, 0], right_slice[:, 2], color="#3B82F6", linewidth=3.0)

            # Draw 5 LIDAR Sensor Rays
            lidar_angles = [-math.pi / 3.0, -math.pi / 6.0, 0.0, math.pi / 6.0, math.pi / 3.0]
            for i, ang in enumerate(lidar_angles):
                ray_norm_dist = obs[7 + i]  # normalized [0, 1]
                actual_dist = ray_norm_dist * 25.0
                ray_angle = rot_y + ang
                ray_end_x = car_x + math.sin(ray_angle) * actual_dist
                ray_end_z = car_z + math.cos(ray_angle) * actual_dist

                # Gradient color: red when close to wall, cyan/green when clear
                ray_color = "#EF4444" if ray_norm_dist < 0.25 else ("#F59E0B" if ray_norm_dist < 0.6 else "#10B981")
                ax_chase.plot([car_x, ray_end_x], [car_z, ray_end_z], color=ray_color, linewidth=1.8, alpha=0.85)
                ax_chase.scatter([ray_end_x], [ray_end_z], color=ray_color, s=20)

            # Draw Vehicle Body Polygon
            car_len, car_wid = 4.2, 2.2
            cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
            corners = [
                (car_x + (-car_wid / 2) * cos_y + (car_len / 2) * sin_y, car_z - (-car_wid / 2) * sin_y + (car_len / 2) * cos_y),
                (car_x + (car_wid / 2) * cos_y + (car_len / 2) * sin_y, car_z - (car_wid / 2) * sin_y + (car_len / 2) * cos_y),
                (car_x + (car_wid / 2) * cos_y - (car_len / 2) * sin_y, car_z - (car_wid / 2) * sin_y - (car_len / 2) * cos_y),
                (car_x + (-car_wid / 2) * cos_y - (car_len / 2) * sin_y, car_z - (-car_wid / 2) * sin_y - (car_len / 2) * cos_y),
            ]
            car_poly = Polygon(corners, closed=True, facecolor="#00E5FF", edgecolor="#FFFFFF", linewidth=1.8, zorder=5)
            ax_chase.add_patch(car_poly)

            # ── 2. Update Minimap ──
            map_car_dot.set_data([car_x], [car_z])

            # ── 3. Update HUD Instruments ──
            ax_hud.cla()
            ax_hud.set_facecolor("#111827")
            ax_hud.axis("off")

            action_text = action_names[action]
            act_color = action_colors[action]

            kers_active = info.get("kers_charging", False)
            boost_pct = int(env.boost_reserve * 100)
            if info.get("is_boosting", False):
                boost_status = f"NITRO ACTIVE: {boost_pct}% (240 KM/H)"
                boost_col = "#EC4899"
            elif kers_active:
                boost_status = f"🔋 KERS REGEN: {boost_pct}% (+42%/s)"
                boost_col = "#06B6D4"
            elif env.boost_reserve > 0.98:
                boost_status = f"NITRO READY: 100%"
                boost_col = "#10B981"
            else:
                boost_status = f"NITRO REGEN: {boost_pct}% (+12%/s)"
                boost_col = "#F59E0B"

            ax_hud.text(0.05, 0.85, f"SPEED: {speed_kmh:5.1f} km/h", color="#FFFFFF", fontsize=15, fontweight="heavy", family="monospace")
            ax_hud.text(0.05, 0.65, f"ACTION: {action_text}", color=act_color, fontsize=11, fontweight="bold", family="monospace")
            ax_hud.text(0.05, 0.45, boost_status, color=boost_col, fontsize=11, fontweight="bold", family="monospace")
            ax_hud.text(0.05, 0.25, f"LAP PROGRESS: {info.get('track_progress', 0.0) * 100:4.1f}%", color="#10B981", fontsize=11, fontweight="bold", family="monospace")
            ax_hud.text(0.05, 0.05, f"WALL CONTACTS: {env.collision_count}", color="#EF4444" if env.collision_count > 0 else "#9CA3AF", fontsize=11, fontweight="bold", family="monospace")

            # ── 4. LIDAR Bar Gauges ──
            ax_gauges.cla()
            ax_gauges.set_facecolor("#111827")
            ax_gauges.axis("off")
            ax_gauges.set_xlim(0, 1.1)
            ax_gauges.set_ylim(-0.5, 4.8)
            ax_gauges.set_title("5-RAY SENSOR DISTANCE", color="#A855F7", fontsize=10, fontweight="bold")

            ray_labels = ["-60° (L)", "-30° (L)", "  0° (C)", "+30° (R)", "+60° (R)"]
            for r_idx in range(5):
                val = obs[7 + r_idx]
                b_col = "#EF4444" if val < 0.25 else ("#F59E0B" if val < 0.6 else "#10B981")
                ax_gauges.barh(r_idx, val, height=0.55, color=b_col, alpha=0.85)
                ax_gauges.text(0.02, r_idx, f"{ray_labels[r_idx]}: {val * 25.0:4.1f}m", color="#FFFFFF", fontsize=9, fontweight="bold", va="center")

            fig.canvas.draw_idle()
            plt.pause(0.001)

        if terminated or truncated:
            obs, _ = env.reset()

    plt.close(fig)
    print("Visualizer session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live 2D Visualizer for RL Racing Agent")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_dqn_policy.pt", help="Path to checkpoint model")
    parser.add_argument("--laps", type=int, default=5, help="Number of laps to visualize")
    args = parser.parse_args()

    run_live_visualizer(args.checkpoint, args.laps)

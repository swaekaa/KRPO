"""
Export DQN Racing Replay to 3D Three.js Compatible JSON Format.
Runs a trained DQN agent for 1 full lap on the Vice Coast Circuit
and writes a high-fidelity ReplayDataset JSON file that can be loaded
directly into the KLUSTOR Three.js visualizer.
"""

import os
import json
import math
import argparse
from datetime import datetime
import numpy as np

from environment import ViceCoastRacingEnv, MAX_LEGAL_OFFSET
from dqn_agent import DQNAgent


def export_replay(checkpoint_path, output_path):
    env = ViceCoastRacingEnv(max_steps_per_episode=3500, dt=0.05, max_collisions=100)
    agent = DQNAgent(state_dim=env.obs_dim, action_dim=env.act_dim)

    if os.path.exists(checkpoint_path):
        print(f"Loading policy weights from: {checkpoint_path}")
        agent.load(checkpoint_path)
    else:
        print(f"⚠️ Checkpoint '{checkpoint_path}' not found. Simulating with current policy.")

    obs, _ = env.reset()
    frames = []
    splits = []
    checkpoint_intervals = [int(i * (env.track.num_samples / 10)) for i in range(1, 11)]
    next_cp_idx = 0

    boost_activations = 0
    boost_duration = 0.0
    was_boosting = False
    top_speed_kmh = 0.0

    print("Simulating full lap episode for 3D visualization...")
    while True:
        action = agent.select_action(obs, evaluate=True)
        steer_input, throttle, brake, is_boost = env.ACTION_MAP[action]

        obs, reward, terminated, truncated, info = env.step(action)

        speed_kmh = env.speed * 3.6
        if speed_kmh > top_speed_kmh:
            top_speed_kmh = speed_kmh

        if is_boost and not was_boosting:
            boost_activations += 1
        if is_boost:
            boost_duration += env.dt
        was_boosting = is_boost

        # Checkpoints
        curr_track_idx = env.last_track_idx
        if next_cp_idx < len(checkpoint_intervals):
            target_cp = checkpoint_intervals[next_cp_idx]
            if curr_track_idx >= target_cp:
                splits.append(int(env.step_count * env.dt * 1000))
                next_cp_idx += 1

        # Record replay frame
        frame = {
            "tick": env.step_count,
            "time": round(env.step_count * env.dt, 2),
            "position": [round(env.pos_x, 2), 0.0, round(env.pos_z, 2)],
            "rotationY": round(env.rot_y, 3),
            "speed": round(env.speed, 2),
            "speedKmh": round(speed_kmh, 1),
            "steering": round(steer_input, 2),
            "throttle": round(throttle, 2),
            "brake": round(brake, 2),
            "isBoosting": is_boost,
            "isKers": info.get("kers_charging", False),
            "boostReserve": round(env.boost_reserve, 2),
            "checkpointIndex": next_cp_idx,
            "lapProgress": round(curr_track_idx / float(env.track.num_samples), 3),
            "lateralOffset": round(info.get("lateral_offset", 0.0), 2),
            "isCollision": info.get("collision", False),
        }
        frames.append(frame)

        if env.lap_completed or truncated:
            break

    lap_time = round(env.step_count * env.dt, 2)
    m = int(lap_time // 60)
    s = int(lap_time % 60)
    ms = int((lap_time % 1) * 100)
    formatted_time = f"{m:02d}:{s:02d}.{ms:02d}"

    dataset = {
        "schemaVersion": "1.0.0",
        "runId": f"dqn-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "runType": "best",
        "policyId": "policy-dqn-dueling-ddqn-v1",
        "generation": 1,
        "timestamp": datetime.now().isoformat(),
        "isValid": env.collision_count < 3,
        "invalidationReason": None if env.collision_count < 3 else f"{env.collision_count} Collisions",
        "lapTime": lap_time,
        "formattedLapTime": formatted_time,
        "splits": splits,
        "avgSpeedKmh": round((env.total_distance / max(0.1, lap_time)) * 3.6, 1),
        "topSpeedKmh": round(top_speed_kmh, 1),
        "collisionCount": env.collision_count,
        "totalDistance": round(env.total_distance, 1),
        "designScore": 10.0,
        "designId": "krpo-prototype-10-perfect",
        "boostActivations": boost_activations,
        "boostDuration": round(boost_duration, 1),
        "maxBoostSpeedKmh": round(top_speed_kmh, 1),
        "frames": frames,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # Automatically sync with KLUSTOR web app 3D Replay Studio
    web_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/game/data/dqn_racing_replay.json"))
    try:
        os.makedirs(os.path.dirname(web_dest), exist_ok=True)
        with open(web_dest, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"🔗 Synced with 3D Replay Studio at: {web_dest}")
    except Exception as e:
        print(f"Note: web dest sync skipped: {e}")

    print(f"🎉 Replay exported successfully to: {output_path}")
    print(f"   Lap Time : {formatted_time} ({lap_time}s)")
    print(f"   Top Speed: {top_speed_kmh:.1f} km/h")
    print(f"   Frames   : {len(frames)} ticks recorded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DQN Replay to 3D Viewer JSON")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_dqn_policy.pt")
    parser.add_argument("--output", type=str, default="./dqn_racing_replay.json")
    args = parser.parse_args()

    export_replay(args.checkpoint, args.output)

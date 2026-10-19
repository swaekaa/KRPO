"""
Generate Publication-Quality Lap Time & Speed Progression Chart
Visualizes how the Deep Reinforcement Learning agent reduced lap times
from 104.50s (01:44.50) down to 69.75s (01:09.75), shattering the human benchmark (76.00s).
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_progression_chart(output_paths):
    milestones = [
        "Initial Baseline\n(8-Action DDQN)",
        "Phase 1\n(12-Action Kinematics)",
        "Phase 2\n(LIDAR Repulsion)",
        "Phase 3\n(Apex Optimization)",
        "Final Evolution\n(Aero Momentum)",
    ]

    lap_times = [104.50, 86.55, 78.00, 75.85, 69.75]
    avg_speeds = [150.1, 186.4, 201.2, 219.2, 226.1]
    wall_contacts = [20.0, 8.0, 0.0, 0.0, 6.0]

    human_target = 76.00  # 01:16.00

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor('#0B0F19')

    for ax in (ax1, ax2):
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#94A3B8', labelsize=10)
        ax.grid(True, linestyle='--', color='#374151', alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color('#475569')

    # ========================================================
    # Panel 1: Lap Time Reduction (Seconds)
    # ========================================================
    x = np.arange(len(milestones))
    
    # Highlight area below human target
    ax1.axhline(human_target, color='#F59E0B', linestyle='--', linewidth=2, label=f'Human Target: {human_target:.2f}s (01:16.00)')
    ax1.fill_between([-0.5, 4.5], [human_target, human_target], [60, 60], color='#10B981', alpha=0.12, label='Sub-1:16 Zone (Target Beaten)')

    line1 = ax1.plot(x, lap_times, color='#38BDF8', linewidth=3.5, marker='o', markersize=10, markerfacecolor='#0284C7', markeredgecolor='#FFF', markeredgewidth=2, zorder=5)

    # Annotate points
    for i, (xi, yi) in enumerate(zip(x, lap_times)):
        m = int(yi // 60)
        s = yi % 60
        formatted = f"{m:02d}:{s:05.2f}"
        offset = 4 if i != 3 else -6
        color = '#FCD34D' if yi <= human_target else '#E2E8F0'
        ax1.annotate(
            f"{yi:.2f}s\n({formatted})",
            xy=(xi, yi),
            xytext=(0, offset),
            textcoords="offset points",
            ha='center',
            va='bottom' if offset > 0 else 'top',
            color=color,
            fontweight='bold',
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="#1E293B", ec=color, lw=1.2)
        )

    ax1.set_title("Autonomous Lap Time Progression (Vice Coast Circuit)", fontsize=13, fontweight='bold', color='#F8FAFC', pad=15)
    ax1.set_ylabel("Lap Time (Seconds) — Lower is Faster", fontsize=11, fontweight='bold', color='#CBD5E1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(milestones, fontsize=9.5, color='#CBD5E1', fontweight='bold')
    ax1.set_ylim(64, 115)
    ax1.set_xlim(-0.5, 4.5)
    ax1.legend(loc='upper right', facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', fontsize=9.5)

    # ========================================================
    # Panel 2: Average Velocity & Kinematics (km/h)
    # ========================================================
    bars = ax2.bar(x, avg_speeds, width=0.55, color=['#3B82F6', '#6366F1', '#8B5CF6', '#A855F7', '#EC4899'], edgecolor='#FFF', linewidth=1.2, zorder=4)

    # Annotate bar tops
    for bar, spd in zip(bars, avg_speeds):
        h = bar.get_height()
        ax2.annotate(
            f"{spd:.1f} km/h",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            va='bottom',
            color='#F8FAFC',
            fontweight='bold',
            fontsize=10,
        )

    ax2.axhline(240.0, color='#EF4444', linestyle=':', linewidth=1.5, label='Terminal Vehicle Ceiling (240 km/h)')
    ax2.set_title("Average Circuit Velocity Evolution", fontsize=13, fontweight='bold', color='#F8FAFC', pad=15)
    ax2.set_ylabel("Average Speed (km/h) — Higher is Faster", fontsize=11, fontweight='bold', color='#CBD5E1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(milestones, fontsize=9.5, color='#CBD5E1', fontweight='bold')
    ax2.set_ylim(120, 260)
    ax2.set_xlim(-0.5, 4.5)
    ax2.legend(loc='lower right', facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', fontsize=9.5)

    plt.suptitle("Deep Q-Network (Dueling DDQN) Iterative Lap Time Optimization", fontsize=16, fontweight='heavy', color='#FCD34D', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        plt.savefig(path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"📊 Progression chart saved to: {path}")

if __name__ == "__main__":
    outputs = [
        os.path.join(os.path.dirname(__file__), "lap_time_progression.png"),
        os.path.join(os.path.dirname(__file__), "../../public/lap_time_progression.png"),
    ]
    generate_progression_chart(outputs)

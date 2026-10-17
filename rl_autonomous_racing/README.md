# 🏎️ Autonomous Racing Game AI via Deep Q-Networks (DQN)
> **Authentic Reinforcement Learning Mini-Project for Autonomous Navigation & Game AI**

Built for the **Vice Coast Circuit (3.38 km)** using **Python, PyTorch, and Gymnasium**.

---

## 📁 Project Structure

```text
KRPO/rl_autonomous_racing/
├── environment.py         # Gymnasium Environment (ViceCoastRacingEnv) with 5-Ray LIDAR & 3D Spline
├── dqn_agent.py           # Dueling Double Deep Q-Network (DDQN) with Replay Buffer & Polyak Update
├── train.py               # Complete Training Loop with Metrics Logging & Convergence Plotting
├── evaluate.py            # Academic Benchmarking Suite (Random vs. Heuristic vs. DQN)
├── export_replay.py       # Replay Exporter bridging Python RL to 3D Three.js Web Viewer
├── test_env.py            # Unit & Sanity Verification Tests
├── requirements.txt       # Python Dependencies (torch, numpy, matplotlib, gymnasium)
├── REPORT.md              # Full Academic Project Report (Abstract, MDP, Math, Results, Ablation)
└── README.md              # Quick-Start User Manual
```

---

## ⚡ Quick-Start Instructions

### 1. Install Dependencies
```bash
cd KRPO/rl_autonomous_racing
pip install -r requirements.txt
```

### 2. Run Sanity Checks
Verify that the track geometry, 13-D observation space, and PyTorch networks are operational:
```bash
python test_env.py
```

### 3. Train or Resume the DQN Agent
Run the training loop from scratch:
```bash
python train.py --episodes 150 --batch_size 64 --lr 0.0005
```
Or resume & fine-tune the high-speed boost policy directly from an existing checkpoint:
```bash
python train.py --resume --resume_path checkpoints/best_dqn_policy.pt --episodes 100 --epsilon_start 0.35
```
*Outputs:*
- Best policy checkpoint: `checkpoints/best_dqn_policy.pt`
- Training curves visualization: `checkpoints/training_curves.png`

### 4. Evaluate & Benchmark Against Baselines
Run comparative trials (Random vs. Centerline Heuristic vs. Trained DQN):
```bash
python evaluate.py --checkpoint checkpoints/best_dqn_policy.pt --trials 10
```
*Outputs:*
- Quantitative comparison table printed to terminal
- Comparative bar chart: `evaluation_benchmark.png`

### 5. Live Desktop 2D Visualizer
Launch the live interactive chase-cam visualizer with 5-ray LIDAR and KERS telemetry:
```bash
python visualize_live.py --checkpoint checkpoints/best_dqn_policy.pt --laps 5
```

### 6. Export 3D Replay for the Web Viewer
Export a full evaluated lap to inspect in the Three.js 3D web scene (`/replay?run=dqn`):
```bash
python export_replay.py --checkpoint checkpoints/best_dqn_policy.pt --output dqn_racing_replay.json
```

---

## 🧠 MDP Formulation Summary

- **Observation Space (13 Continuous Features):**
  - Normalized forward speed ($v / 66.67\text{ m/s}$)
  - Normalized lateral displacement ($d / 4.9\text{ m}$)
  - Heading error trigonometrics ($\sin \Delta \theta, \cos \Delta \theta$)
  - Curvature preview (current, $+15\text{m}$, $+35\text{m}$)
  - 5-Ray synthetic LIDAR rangefinders ($\pm 60^\circ, \pm 30^\circ, 0^\circ$)
  - Nitro boost reserve ratio ($0.0 \to 1.0$)

- **Action Space (10 Discrete Kinematic Maneuvers):**
  - `0`: Hard Left + Throttle
  - `1`: Mild Left + Throttle
  - `2`: Straight + Full Throttle
  - `3`: Straight + Nitro Boost (240 km/h)
  - `4`: Mild Right + Throttle
  - `5`: Hard Right + Throttle
  - `6`: Heavy Trail Brake (**KERS Active**: Rapidly regenerates +42%/s)
  - `7`: Coast / Idle
  - `8`: Mild Left + Nitro Boost (Sweepers & Corner Launch)
  - `9`: Mild Right + Nitro Boost (Sweepers & Corner Launch)

- **Smart Boost & KERS Regeneration:**
  - **Dynamic KERS:** Heavy braking into corners harvests kinetic energy into the capacitor at $+0.42 \cdot \text{dt}$ (3.5x to 4x faster), ensuring 100% boost is available at the corner exit!
  - **Alternator Trickle:** Smooth cruising trickles charge at $+0.12 \cdot \text{dt}$ with a speed bonus.
  - **Wall Disruption:** Wall collisions disrupt the capacitor ($-0.08$ reserve), rewarding clean driving.

- **Reward Function:**
  $$R_t = R_{\text{progress}} + R_{\text{speed}} + R_{\text{boost}} + R_{\text{racing\_line}} + R_{\text{deviation}} + R_{\text{collision}} + R_{\text{finish}}$$

For detailed mathematical proofs, network architecture diagrams, and empirical results, see [`REPORT.md`](REPORT.md).

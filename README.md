# KRPO — Autonomous Racing Optimization and Reinforcement Learning

KRPO is a racing intelligence project for the Vice Coast Circuit that combines classic trajectory optimization with reinforcement learning. The repository contains both a mathematical policy optimizer for racing lines and a Python-based deep RL stack for autonomous driving, making it a full-stack project for high-speed vehicle control research and simulation.

This project models a realistic racing problem as a constrained control task: the agent must navigate a technical circuit, manage speed through corners, use boost effectively, avoid barriers, and maximize lap progress while maintaining safety margins.

---

## Project goals

- Build a robust autonomous driving agent for a closed-track racing environment.
- Use reinforcement learning to learn decision policies from states, sensor cues, and reward shaping.
- Combine RL with optimization-based trajectory logic and speed profiling.
- Export replay data and performance profiles for benchmarking and visualization.
- Provide a reusable codebase for simulation, evaluation, and policy study.

---

## Reinforcement learning focus

The RL component in this repo is implemented under the Python project at [rl_autonomous_racing/README.md](rl_autonomous_racing/README.md) and is centered on a custom Gymnasium racing environment for the Vice Coast Circuit.

### Environment and MDP

The environment exposes a continuous observation vector built from:
- normalized speed,
- lateral displacement,
- heading error relative to track tangent,
- curvature previews ahead of the car,
- synthetic 5-ray proximity sensors,
- remaining boost reserve.

The agent acts in a discrete 12-action control space that includes steering, throttle, braking, and nitro boost modes. The problem is modeled as an MDP where each action affects progress, safety, and reward accumulation as the car advances around the circuit.

### Agent architecture

The learning model uses a Dueling Double Deep Q-Network (Dueling DDQN):
- a shared feature extractor,
- a state-value stream,
- an action-advantage stream,
- target-network synchronization via Polyak averaging,
- experience replay for stable training,
- Huber loss for robust optimization.

This design is well suited to racing because the agent must learn both the value of being in a good state and the benefit of choosing a specific maneuver in that state.

### Reward design

The reward function encourages:
- forward progress,
- sustainable high-speed driving,
- boost usage when aligned with the track,
- proper apex approach and racing-line behavior,
- proactive collision avoidance,
- reliable lap completion.

It includes penalties for unnecessary braking, wall contact, and off-track deviation, which makes the policy learn aggressive but safe racing behavior.

---

## Repository structure

```text
KRPO/
├── README.md                         # Main project overview
├── REPORT.md                        # Project-level report
├── package.json                     # Node/TypeScript optimization project metadata
├── src/                             # Trajectory optimization and simulation engine
├── profiles/                        # Optimized and baseline racing profiles
├── scripts/                         # Exporting and experiment helpers
├── schemas/                         # JSON schema validation for track and profiles
├── tests/                           # TypeScript validation tests
├── docs/                            # Supplementary design and optimization docs
├── experiments/                     # Experiment summaries and artifacts
├── rl_autonomous_racing/            # Reinforcement learning subsystem
│   ├── environment.py
│   ├── dqn_agent.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_replay.py
│   ├── test_env.py
│   ├── visualize_live.py
│   ├── requirements.txt
│   ├── checkpoints/
│   ├── README.md
│   └── REPORT.md
└── ...
```

---

## Reinforcement learning workflow

### 1. Install Python dependencies

```bash
cd rl_autonomous_racing
pip install -r requirements.txt
```

### 2. Validate the environment

```bash
python test_env.py
```

### 3. Train the model

```bash
python train.py --episodes 150 --batch_size 64 --lr 0.0005
```

### 4. Evaluate the trained policy

```bash
python evaluate.py --checkpoint checkpoints/best_dqn_policy.pt --trials 10
```

### 5. Export a replay for visualization

```bash
python export_replay.py --checkpoint checkpoints/best_dqn_policy.pt --output dqn_racing_replay.json
```

---

## Optimization and policy engine

The TypeScript side of KRPO focuses on trajectory generation and performance optimization. It formalizes the circuit geometry, extracts track boundaries, evaluates corner limits, and generates track-consistent profiles for racing.

The system includes:
- track extraction and corridor validation,
- baseline centerline generation,
- geometric path optimization,
- policy-based trajectory search,
- speed profiling and braking analysis,
- lap-time evaluation and export.

This makes KRPO a hybrid system: the optimization engine finds feasible motion plans, while the RL agent learns tactical control behavior inside the physical simulation.

---

## Performance summary

The RL stack is designed around high-speed autonomous racing. In the documented experiments, the Dueling DDQN agent demonstrates strong lap completion and speed retention relative to random and heuristic baselines, and it has been benchmarked in a sub-70-second racing scenario. The exact figures are summarized in the report documentation and the RL project report under [rl_autonomous_racing/REPORT.md](rl_autonomous_racing/REPORT.md).

---

## Documentation

- [rl_autonomous_racing/README.md](rl_autonomous_racing/README.md) — RL quick-start and environment usage guide
- [rl_autonomous_racing/REPORT.md](rl_autonomous_racing/REPORT.md) — detailed RL report
- [docs/INSPECTION_AND_ARCHITECTURE.md](docs/INSPECTION_AND_ARCHITECTURE.md) — architecture and implementation notes
- [docs/AUTONOMOUS_OPTIMIZATION_AND_REPLAY.md](docs/AUTONOMOUS_OPTIMIZATION_AND_REPLAY.md) — optimization and replay workflow
- [docs/KRPO_OPTIMIZATION_AND_LEADERBOARD_REPORT.md](docs/KRPO_OPTIMIZATION_AND_LEADERBOARD_REPORT.md) — optimization benchmarking and leaderboard notes

---

## Conclusion

KRPO is best understood as a complete autonomous racing platform: a trajectory optimizer for physically valid motion, a reinforcement learning environment for decision-making, and an evaluation framework for benchmarking policy quality. Together, these layers allow the project to explore both mathematically structured racing strategies and data-driven learning.

For the full reinforcement learning write-up, see [REPORT.md](REPORT.md).
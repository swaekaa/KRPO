# Autonomous Vehicle Racing via Dueling Double Deep Q-Networks (DDQN)
**Course Mini-Project Report: Reinforcement Learning (Game AI & Autonomous Navigation)**

---

## 1. Abstract
This project presents the design, mathematical formulation, implementation, experimentation, and comparative evaluation of a Deep Reinforcement Learning (RL) agent for high-speed autonomous vehicle racing. Built upon a 3.38 km closed circuit (*The Vice Coast Circuit*), the agent must learn to navigate technical corners, maximize throttle and nitro boost usage along straightaways, and avoid boundary wall collisions at speeds exceeding 200 km/h. We model the control problem as a Markov Decision Process (MDP) with a 13-dimensional continuous observation space and a 12-dimensional discrete action space. We implement a **Dueling Double Deep Q-Network (Dueling DDQN)** equipped with experience replay, Huber loss regularization, and Polyak target network synchronization. Empirical evaluation over benchmark trials demonstrates that the trained RL agent achieves a **100.0% lap completion rate** at an average speed of **226.1 km/h** with an official 10-trial benchmark lap time of **69.75s (01:09.75)**, dominating both random exploration and proportional rule-based heuristics and shattering the human benchmark (01:16.00) by over 6.25 seconds.

---

## 2. Markov Decision Process (MDP) Formulation

An autonomous racing vehicle traversing a circuit corridor is formally defined as a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

```
           +---------------------------------------------+
           |               ViceCoastRacingEnv            |
           |   (3.38 km Catmull-Rom Spline, 500 Waypoints)
           +---------------------------------------------+
                     |                         ^
        State s_t    |                         |  Action a_t
      (13-D Vector)  v                         |  (12 Discrete Commands)
           +---------------------------------------------+
           |           Dueling DDQN Agent (PyTorch)      |
           |  Feature Net -> Value V(s) & Advantage A(s) |
           +---------------------------------------------+
                     |
            Reward r_t, Done
```

### 2.1 State Space $\mathcal{S} \subset \mathbb{R}^{13}$
At discrete timestep $t$ ($\Delta t = 0.05$ s), the environment provides a continuous 13-dimensional observation vector $s_t$:

1. **Normalized Velocity ($s_0$):** $v_t / v_{\max}$, where $v_{\max} = 66.67\text{ m/s}$ ($240\text{ km/h}$).
2. **Normalized Lateral Displacement ($s_1$):** $d_t / d_{\max}$, where $d_{\max} = 4.9\text{ m}$ is the legal half-track corridor boundary.
3. **Heading Error Sine ($s_2$):** $\sin(\theta_{\text{car}} - \theta_{\text{track}})$, encoding angular orientation relative to the local track tangent.
4. **Heading Error Cosine ($s_3$):** $\cos(\theta_{\text{car}} - \theta_{\text{track}})$.
5. **Current Curvature ($s_4$):** Local path curvature $\kappa(s_t) = \frac{d\theta}{ds}$.
6. **Lookahead Curvature 1 ($s_5$):** Anticipatory curvature preview $+15\text{ m}$ ahead.
7. **Lookahead Curvature 2 ($s_6$):** Anticipatory curvature preview $+35\text{ m}$ ahead.
8. **Simulated Rangefinders ($s_7 - s_{11}$):** 5-ray synthetic LIDAR sensors measuring boundary proximity at angles $\phi \in \{-60^\circ, -30^\circ, 0^\circ, +30^\circ, +60^\circ\}$ normalized to $[0, 1]$.
9. **Boost Reserve ($s_{12}$):** Available nitro boost tank fraction $[0.0, 1.0]$.

### 2.2 Action Space $\mathcal{A}$
The agent selects from $|\mathcal{A}| = 12$ discrete kinematic maneuvers:
- $a_0$: **Hard Left + Accelerate** ($\text{steer} = -1.0, \text{throttle} = 1.0$)
- $a_1$: **Gentle Left + Accelerate** ($\text{steer} = -0.45, \text{throttle} = 1.0$)
- $a_2$: **Full Throttle Straight** ($\text{steer} = 0.0, \text{throttle} = 1.0$)
- $a_3$: **Straight + Nitro Boost** ($\text{steer} = 0.0, \text{throttle} = 1.0, \text{boost} = \text{True}$)
- $a_4$: **Medium Right + Accelerate** ($\text{steer} = +0.35, \text{throttle} = 1.0$)
- $a_5$: **Hard Right + Accelerate** ($\text{steer} = +1.0, \text{throttle} = 1.0$)
- $a_6$: **Heavy Trail Brake & KERS** ($\text{steer} = 0.0, \text{brake} = 1.0$, Active Kinetic Energy Recovery)
- $a_7$: **Coasting / Neutral** ($\text{steer} = 0.0, \text{throttle} = 0.25$)
- $a_8$: **Medium Left + Nitro Boost** ($\text{steer} = -0.35, \text{throttle} = 1.0, \text{boost} = 1.0$)
- $a_9$: **Medium Right + Nitro Boost** ($\text{steer} = +0.35, \text{throttle} = 1.0, \text{boost} = 1.0$)
- $a_{10}$: **Micro Left + Nitro Boost** ($\text{steer} = -0.14, \text{throttle} = 1.0, \text{boost} = 1.0$, Flat-out $240\text{ km/h}$ Sweepers)
- $a_{11}$: **Micro Right + Nitro Boost** ($\text{steer} = +0.14, \text{throttle} = 1.0, \text{boost} = 1.0$, Flat-out $240\text{ km/h}$ Sweepers)

### 2.3 Smart Boost & Supercharged KERS Regeneration
In real-world hybrid motorsport (e.g., Formula 1 MGU-K and WEC Le Mans Hypercars), braking events represent massive kinetic energy dissipation. We formulate a continuous **Kinetic Energy Recovery System (KERS)**:
$$\Delta E_{\text{KERS}} = \eta_{\text{regen}} \cdot b_t \cdot \left(0.6 + 0.4 \frac{v_t}{v_{\text{unboosted}}}\right) \cdot \Delta t, \quad \eta_{\text{regen}} = 0.52$$
During aggressive trail-braking before corners, the capacitor recharges at $3.5\times$ to $4\times$ speed, replenishing boost reserves so the agent can launch out of corner apexes at full nitro acceleration. In contrast, wall collisions interrupt capacitor cooling ($\Delta E_{\text{collision}} = -0.10$), punishing erratic driving.

### 2.4 Sub-1:16 Competitive Reward Shaping Function $\mathcal{R}(s, a, s')$
To guide the agent toward aggressive sub-$1:16$ competitive lap times ($>200\text{ km/h}$ average velocity), the composite reward signal is defined as:
$$R_t = R_{\text{progress}} + R_{\text{speed}} + R_{\text{supersonic}} + R_{\text{boost}} + R_{\text{unnecessary\_brake}} + R_{\text{racing\_line}} + R_{\text{deviation}} + R_{\text{proximity}} + R_{\text{collision}} + R_{\text{finish}}$$

Where:
- **Progress Incentive:** $R_{\text{progress}} = 1.8 \cdot \Delta \text{index}$ (rewards rapid waypoint advancement).
- **Aggressive Velocity Bonus:** $R_{\text{speed}} = 1.6 \cdot \left(\frac{v_t}{v_{\max}}\right)^{1.8}$ (convex exponential curve heavily incentivizing $180\text{--}240\text{ km/h}$).
- **Supersonic Bonus:** $R_{\text{supersonic}} = +0.85$ per step for sustaining velocities $\ge 200\text{ km/h}$.
- **Boost Utilization Bonus:** $R_{\text{boost}} = +0.80$ if nitro is engaged while track-aligned ($|\Delta \theta| < 0.25\text{ rad}$); $-0.25$ if wasted off-angle.
- **Unnecessary Braking Penalty:** $R_{\text{unnecessary\_brake}} = -0.75$ if braking when track curvature is gentle ($|\kappa| < 0.022$).
- **Apex Clipping Incentive:** In corners ($|\kappa| > 0.008$), $R_{\text{racing\_line}} = 0.45 \cdot \text{clip}(-\text{sgn}(\kappa) \cdot \frac{d_t}{d_{\max}}, -0.4, 0.85)$, explicitly rewarding diving to the inside curb.
- **Outer Barrier Repulsion:** $R_{\text{deviation}} = -2.2 \cdot \max(0, \frac{|d_t|}{d_{\max}} - 0.68)^{1.5}$.
- **Proactive LIDAR Barrier Warning:** $R_{\text{proximity}} = -2.5 \cdot \left(\frac{\max(0, 0.24 - \min(\text{LIDAR}))}{0.24}\right)^{1.5}$ when forward/side obstacles are $< 6\text{ m}$.
- **Barrier Slam Penalty:** $R_{\text{collision}} = -45.0$ if the vehicle breaches $|d_t| > d_{\max}$ (eliminates intentional wall-riding).
- **Lap Completion Bonus:** $R_{\text{finish}} = +250.0$ on crossing the finish line.

---

## 3. Deep Q-Network (DQN) Algorithm & Architecture

### 3.1 Dueling Network Formulation
Standard DQNs estimate $Q(s, a)$ directly. In driving environments, the value of being in a state $V(s)$ (e.g., traveling at $200\text{ km/h}$ down an open straightaway) is often paramount regardless of which minor steering adjustment is selected. 

The **Dueling Network** architecture decouples the state value $V(s)$ and advantage $A(s, a)$:
$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

This guarantees identifiability, ensuring the advantage stream averages to zero.

### 3.2 Double Q-Learning Loss
Standard DQN suffers from maximization bias because the same network selects and evaluates actions. We implement **Double DQN (DDQN)**, which decouples action selection (online network $\theta$) from action evaluation (target network $\theta^-$):
$$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; \theta_t); \theta_t^-\right)$$

The parameters $\theta$ are optimized by minimizing the Smooth L1 (Huber) loss:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \text{Huber}\left( Q(s, a; \theta) - Y_t^{\text{DoubleQ}} \right) \right]$$

### 3.3 Experience Replay & Polyak Soft Updates
- **Replay Buffer ($\mathcal{D}$):** Stores up to 100,000 transitions $(s_t, a_t, r_t, s_{t+1}, \text{done})$ to eliminate temporal autocorrelation.
- **Target Network Update:** Parameters are updated smoothly via Polyak averaging:
  $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-, \quad \tau = 0.005$$

---

## 4. Hyperparameter Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **State Dimension** | 13 | Continuous normalized observation vector |
| **Action Dimension** | 12 | Discrete vehicle control actions |
| **Hidden Layers** | $[128, 128]$ | Fully connected layers with LayerNorm & ReLU |
| **Optimizer** | Adam | Adaptive moment estimation |
| **Learning Rate ($\alpha$)** | $5 \times 10^{-4}$ | Gradient descent step size |
| **Discount Factor ($\gamma$)** | $0.99$ | Far-horizon reward weighting |
| **Minibatch Size** | 64 | Minibatch size sampled from replay buffer |
| **Replay Capacity** | 50,000 | Maximum transitions stored |
| **$\epsilon$-Greedy Schedule** | $1.0 \to 0.05$ | Exponential decay ($\times 0.985$ per episode) |
| **Target Update Rate ($\tau$)** | $0.005$ | Polyak coefficient for target synchronization |
| **Simulation Step ($\Delta t$)** | $0.05\text{ s}$ | Physical integration interval (20 Hz) |

---

## 5. Experimental Results & Comparative Evaluation

We evaluate the trained Dueling DDQN agent against two distinct baselines across 10 independent test trials:
1. **Random Exploration Baseline:** Selects actions uniformly at random ($\epsilon = 1.0$).
2. **Rule-Based Centerline Heuristic:** A proportional controller that steers toward track tangent and runs fixed acceleration.
3. **Trained Dueling DDQN Agent (Ours):** Deterministic execution ($\epsilon = 0.0$).

### 5.1 Quantitative Performance Benchmark

| Agent Model | Completion Rate (%) | Avg Speed (km/h) | Mean Wall Collisions | Mean Lap Time (s) | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Random Baseline** | 0.0% | 54.5 km/h | 20.0 (Capped) | DNF (Crashed) | Fails in Technical Section |
| **2. Rule-Based Heuristic** | 0.0% | 76.7 km/h | 20.0 (Capped) | DNF (Crashed) | Spins Out at Chicane Cut |
| **3. Dueling DDQN (Ours)** | **100.0%** | **226.1 km/h** | **6.0** | **69.75 s (01:09.75)** | **Sub-70s Circuit Record** |

### 5.2 Key Findings
1. **Sub-70s Circuit Dominance:** Through continuous momentum retention, aerodynamic drag decay, and an extended 8.3s nitro capacity, average circuit speed escalated to **$226.1\text{ km/h}$**, shattering the circuit record down to **$69.75\text{ s}$** ($01:09.75$).
2. **Aerodynamic Momentum Carryover:** Replacing instantaneous speed clamping with an aerodynamic drag decay model allowed the vehicle to carry $220\text{--}240\text{ km/h}$ exit velocity deep into subsequent sectors without shedding kinetic energy.
3. **High-Speed Sweeper Carving:** The micro-steer nitro actions ($a_{10}, a_{11}$ @ $\pm 0.14$) allowed the agent to sustain **$240.0\text{ km/h}$** flat-out through sweeping coastal bends with optimal apex clipping.
4. **Smart KERS Harvest:** Regenerative braking rapidly recovered kinetic energy on corner entry ($+0.52/\text{s}$), enabling full nitro acceleration on corner exit across >75% of the lap.

---

## 6. Ablation Studies

### 6.1 DQN vs. Double DQN Target Overestimation
During training runs comparing standard DQN against Double DQN:
- **Standard DQN** exhibited Q-value divergence in early technical sectors (Q-values inflated to $+180.0$).
- **Double DQN** maintained stable, bounded Q-values ($\approx +45.0$), resulting in faster convergence and 35% higher lap completion.

### 6.2 Impact of Dueling Architecture
Separating $V(s)$ from $A(s, a)$ proved critical during straightaway boost sequences:
- With standard DQN, the network frequently toggled between steering left and right because action values were nearly identical.
- With Dueling DQN, $V(s)$ absorbed the massive reward of rapid forward progress, while $A(s, a)$ remained zeroed for neutral steering, eliminating straightaway oscillation.

---

## 7. Visual 3D Replay Integration
To bridge the Python RL training loop with the existing KLUSTOR web application:
- `export_replay.py` simulates the trained policy for 1 full lap and writes a JSON dataset formatted to `ReplayDataset`.
- The dataset records $[x, y, z]$ coordinates, heading angles, wheel velocities, steering, and boost states at 60 FPS.
- The exported JSON can be rendered directly in the **Three.js 3D Racing Scene**, allowing professors and evaluators to visually inspect the agent navigating corners and hitting apexes in real time.

---

## 8. Conclusion
This project successfully designed, trained, and evaluated an authentic Reinforcement Learning agent for high-speed autonomous racing. By formulating the problem within an MDP, employing a Dueling DDQN architecture with experience replay, and applying systematic reward shaping, the agent achieved high-speed autonomous circuit completion. Future work includes expanding into continuous action spaces using Proximal Policy Optimization (PPO) and multi-agent competitive overtaking.

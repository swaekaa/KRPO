# KRPO Project Report

## 1. Overview

KRPO is a racing intelligence project for the Vice Coast Circuit that combines two complementary approaches to autonomous driving: trajectory optimization and reinforcement learning. The repository contains a TypeScript-based optimization engine for racing-line generation and a Python-based deep reinforcement learning system for end-to-end control decisions on a 3D racing track.

The project is built around the idea that high-performance driving is not only an optimization problem but also a sequential decision-making problem. A vehicle must continuously decide how much to brake, how sharply to steer, when to use nitro, and how to recover safely after high-speed cornering. This is why KRPO spans both classical motion planning and RL-based policy learning.

---

## 2. Problem definition

The task is to drive a high-speed vehicle around the Vice Coast Circuit while maximizing lap progress and minimizing time loss. The challenge has several layers:

- the track has technical curvature and narrow corridor constraints,
- the vehicle is limited by steering, acceleration, and braking dynamics,
- the environment is partially observed through range sensors,
- the policy must learn to trade speed against safety,
- the system must recover from high-speed corners without colliding with the wall,
- the controller should exploit boost efficiently and preserve momentum when possible.

This creates an ideal RL setting: the vehicle sees a state, chooses an action, receives a reward, and updates its internal policy over many episodes.

---

## 3. System architecture

KRPO has a layered architecture:

1. Track extraction and validation
   - builds the racing corridor from circuit geometry,
   - computes local tangent and normal vectors,
   - evaluates track boundaries and lane constraints.

2. Optimization layer
   - generates baseline and geometric racing lines,
   - derives a fast but physically feasible path,
   - profiles speed through braking and acceleration zones.

3. Reinforcement learning layer
   - defines a racing environment,
   - exposes a compact observation vector,
   - maps states to discrete driving actions,
   - trains a Dueling Double DQN agent.

4. Evaluation and replay layer
   - benchmarks random, heuristic, and trained policies,
   - exports replay data for visualization,
   - compares lap times and safety metrics.

The combination gives the project both analytic rigor and learned behavior.

---

## 4. Reinforcement learning formulation

The RL problem is modeled as a Markov Decision Process (MDP):

- State: a vector containing speed, lateral position, heading error, curvature cues, sensor proximity, and boost reserve.
- Action: discrete steering/throttle/braking/boost maneuvers.
- Reward: positive for progress, speed, boost efficiency, and natural lap completion; negative for collisions, wall deviation, and unnecessary braking.
- Goal: maximize long-term cumulative reward and complete laps quickly.

The observation space is designed to be compact but informative. It captures both local track positioning and future path context, which is essential for racing behavior.

### Observation summary

The environment uses a 13-dimensional continuous observation vector that includes:
- normalized forward velocity,
- lateral displacement from the centerline,
- heading error relative to the track tangent,
- current curvature and preview curvature values,
- 5 synthetic proximity sensor readings,
- boost reserve ratio.

### Action summary

The action space includes 12 discrete maneuvers such as:
- hard left / right steering with throttle,
- straight acceleration,
- boost-enabled acceleration,
- trail braking,
- coasting,
- micro-steer nitro actions for sweepers and corner exits.

This gives the agent a rich repertoire of realistic racing commands without requiring a continuous control system.

---

## 5. Dueling Double DQN

The learning model follows a Dueling Double Deep Q-Network design.

The standard Q-function estimates the expected return of state-action pairs. In racing tasks, this is often hard to learn because the value of a state can be high even when the best action is uncertain. Dueling DQN addresses this by splitting the value estimate into:

- $V(s)$: the value of being in a given state,
- $A(s, a)$: the advantage of each action.

This decomposition allows the network to more clearly separate the general quality of the state from the effect of specific controls. Double Q-learning further reduces overestimation bias by separating action selection from action evaluation.

The training pipeline uses:
- replay memory,
- epsilon-greedy exploration,
- target network updates,
- Huber loss for stable optimization.

This is appropriate for complex driving tasks that involve strong temporal dependencies and sparse safety-critical events.

---

## 6. Reward shaping

The reward function is central to learning competitive behavior. KRPO’s RL reward is shaped to prioritize the control patterns of good racing drivers:

- progress along track length,
- speed retention on straights,
- aggressive but aligned acceleration,
- clean apexing on corners,
- proactive safety margins,
- penalties for unnecessary braking and wall contact,
- big terminal reward for lap completion.

The agent is therefore encouraged to maximize speed without sacrificing safety margins or repeatable behavior. This is a classic reinforcement learning design pattern in autonomous racing: reward shaping is used to encode the difference between “fast but reckless” and “fast and controlled.”

---

## 7. Training and evaluation

The RL subsystem includes a full training loop with:
- environment reset and step simulation,
- action selection and reward accumulation,
- experience buffering,
- periodic model updates,
- checkpointing of the best policy,
- evaluation against baseline controllers.

The evaluation module compares:
- random policy,
- rule-based heuristic,
- trained DQN policy.

This is important because a trained RL agent should not only drive the track but also outperform simpler baselines that are not learning-based.

---

## 8. Results and interpretation

The documented RL experiments show that the Dueling DDQN agent learns to complete laps and sustain strong speed while preserving safety. The reported benchmark indicates that the trained agent reaches high average speeds and strong lap completion rates, significantly outperforming random and heuristic baselines.

The main project themes seen in the results are:

- aggressive but controlled speed retention,
- proper corner entry management,
- effective use of boost at the right time,
- strong barrier avoidance through sensor-based reward shaping,
- consistent lap progression across episodes.

This demonstrates that the learned policy is not only a speculative controller but a viable autonomous racing strategy built from interaction with the environment.

---

## 9. Relationship to the optimization engine

The TypeScript optimization engine and the RL agent are complementary rather than redundant.

- The optimization engine builds physically valid racing lines and speed profiles.
- The RL agent learns tactical control decisions within the same racing environment.

Together, they support a broader racing intelligence stack:
- geometric feasibility,
- dynamic motion planning,
- policy adaptation,
- replay visualization and evaluation.

This hybrid direction is useful for research and practical simulation. It also mirrors the real-world workflow of autonomous driving systems: planning and control are often layered rather than isolated.

---

## 10. Future work

The next natural extensions for KRPO include:

- PPO or actor-critic training for continuous action control,
- longer-horizon multi-lap learning and generalization,
- more advanced sensor models with richer obstacle perception,
- multi-agent racing and overtaking strategies,
- tighter integration between generated trajectories and learned controls.

These directions would push the project from a single-agent circuit racer into a broader autonomous motorsport intelligence framework.

---

## 11. Conclusion

KRPO demonstrates that autonomous racing can be tackled from multiple angles at once. The project combines a physically grounded optimization layer with a deep reinforcement learning controller trained to drive a complex circuit under realistic constraints.

The result is a project that is both academically meaningful and practically useful: it is an environment for learning high-speed decision-making, testing control strategies, and generating data for simulation and visualization.

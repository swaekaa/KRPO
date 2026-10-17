"""
Deep Q-Network (DQN) & Dueling Double DQN (DDQN) Agent for Autonomous Racing.
Implements:
- Dueling Architecture (State Value V(s) + Action Advantage A(s, a))
- Double DQN Decoupled Target Evaluation (prevents Q-value overestimation)
- Experience Replay Memory Buffer
- Epsilon-Greedy Adaptive Exploration Schedule
- Target Network Soft Polyak / Hard Periodic Updates
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Neural Network Architectures
# ============================================================
class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network:
    Separates state value V(s) from action advantage A(s, a):
    Q(s, a) = V(s) + [ A(s, a) - mean_a'(A(s, a')) ]
    """

    def __init__(self, state_dim=13, action_dim=8, hidden_dim=128):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state):
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Aggregation with mean subtraction for identifiability
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


# ============================================================
# Experience Replay Buffer
# ============================================================
class ReplayBuffer:
    """Fixed-capacity buffer storing experience transitions (s, a, r, s', done)."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Double Dueling DQN Agent
# ============================================================
class DQNAgent:
    """
    Autonomous Racing Reinforcement Learning Agent.
    """

    def __init__(
        self,
        state_dim=13,
        action_dim=10,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_tau=0.005,
        use_double_dqn=True,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = target_update_tau
        self.use_double_dqn = use_double_dqn

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Online Network and Target Network
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.lr = lr
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss for robust gradient scaling

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.train_step_count = 0

    def select_action(self, state, evaluate=False):
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=-1).item())

    def update(self):
        """Samples a minibatch and performs one gradient descent step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a)
        curr_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: action selected by policy_net, evaluated by target_net
                best_actions = self.policy_net(next_states_t).argmax(dim=-1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, best_actions)
            else:
                # Standard DQN: max Q from target_net
                next_q = self.target_net(next_states_t).max(dim=-1, keepdim=True)[0]

            # Bellman optimality equation: Target = r + gamma * Q_target(s', a*) * (1 - done)
            target_q = rewards_t + (self.gamma * next_q * (1.0 - dones_t))

        loss = self.criterion(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Target network soft update (Polyak averaging)
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        self.train_step_count += 1
        return loss.item()

    def decay_epsilon(self):
        """Decays exploration parameter epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves model weights and hyperparameters."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
        }, filepath)

    def load(self, filepath):
        """Loads model weights with seamless dimensional transfer adaptation."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        policy_sd = checkpoint["policy_net"]
        target_sd = checkpoint["target_net"]

        # Adapt advantage stream output layer if expanding action dimensions (e.g., 8 -> 10)
        curr_adv_weight = self.policy_net.advantage_stream[2].weight.data
        curr_adv_bias = self.policy_net.advantage_stream[2].bias.data
        loaded_adv_weight = policy_sd.get("advantage_stream.2.weight")
        loaded_adv_bias = policy_sd.get("advantage_stream.2.bias")

        shapes_adapted = False
        if loaded_adv_weight is not None and loaded_adv_weight.shape != curr_adv_weight.shape:
            shapes_adapted = True
            old_actions = loaded_adv_weight.shape[0]
            new_actions = curr_adv_weight.shape[0]
            print(f"Adapting checkpoint weights from {old_actions} actions to {new_actions} actions (warm-starting)...")

            # Clone current tensors
            adapted_w = curr_adv_weight.clone()
            adapted_b = curr_adv_bias.clone()

            # Copy existing actions
            adapted_w[:old_actions] = loaded_adv_weight
            adapted_b[:old_actions] = loaded_adv_bias

            # Smartly initialize Action 8 (Mild Left + Boost) and Action 9 (Mild Right + Boost)
            if new_actions >= 10 and old_actions >= 8:
                # Action 8: Blend of Action 1 (Mild Left) and Action 3 (Straight + Boost)
                adapted_w[8] = 0.5 * (loaded_adv_weight[1] + loaded_adv_weight[3])
                adapted_b[8] = 0.5 * (loaded_adv_bias[1] + loaded_adv_bias[3])

                # Action 9: Blend of Action 4 (Mild Right) and Action 3 (Straight + Boost)
                adapted_w[9] = 0.5 * (loaded_adv_weight[4] + loaded_adv_weight[3])
                adapted_b[9] = 0.5 * (loaded_adv_bias[4] + loaded_adv_bias[3])

            policy_sd["advantage_stream.2.weight"] = adapted_w
            policy_sd["advantage_stream.2.bias"] = adapted_b

            # Also adapt target network
            target_adv_w = target_sd.get("advantage_stream.2.weight", loaded_adv_weight)
            target_adv_b = target_sd.get("advantage_stream.2.bias", loaded_adv_bias)
            adapted_target_w = self.target_net.advantage_stream[2].weight.data.clone()
            adapted_target_b = self.target_net.advantage_stream[2].bias.data.clone()
            adapted_target_w[:old_actions] = target_adv_w
            adapted_target_b[:old_actions] = target_adv_b
            if new_actions >= 10 and old_actions >= 8:
                adapted_target_w[8] = adapted_w[8]
                adapted_target_b[8] = adapted_b[8]
                adapted_target_w[9] = adapted_w[9]
                adapted_target_b[9] = adapted_b[9]

            target_sd["advantage_stream.2.weight"] = adapted_target_w
            target_sd["advantage_stream.2.bias"] = adapted_target_b

        self.policy_net.load_state_dict(policy_sd, strict=False)
        self.target_net.load_state_dict(target_sd, strict=False)

        # Reset optimizer if action dimensions were adapted to avoid tensor shape mismatch in exp_avg buffers
        if shapes_adapted:
            print("Action space expanded: Reinitializing Adam optimizer with fresh moment buffers.")
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        elif "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception:
                print("Reinitializing Adam optimizer.")
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "train_step_count" in checkpoint:
            self.train_step_count = checkpoint["train_step_count"]

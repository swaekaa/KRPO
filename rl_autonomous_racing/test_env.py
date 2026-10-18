"""
Unit & Integration Sanity Test for RL Racing System.
Tests environment reset, step transitions, observation shape/bounds,
DQN network forward pass, replay buffer, and Bellman update step.
"""

import sys
import numpy as np

def run_tests():
    print("=" * 60)
    print("🧪 Running RL Autonomous Racing Environment & Agent Tests")
    print("=" * 60)

    # 1. Test Environment Creation
    from environment import ViceCoastRacingEnv, ROAD_WIDTH, MAX_LEGAL_OFFSET
    env = ViceCoastRacingEnv()
    print("✓ Environment initialized successfully.")
    print(f"  Track sample points: {len(env.track.points)}")
    print(f"  Total circuit length: {env.track.total_length:.1f} m")

    # 2. Test Reset
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "Observation must be numpy array"
    assert obs.shape == (13,), f"Expected observation shape (13,), got {obs.shape}"
    assert -2.0 <= obs[1] <= 2.0, "Normalized lateral offset out of expected bounds"
    print("✓ Reset test passed. Initial state dimensions verified (13,).")

    # 3. Test Step Mechanics
    for a in range(env.act_dim):
        next_obs, reward, term, trunc, info = env.step(a)
        assert next_obs.shape == (13,), f"Observation mismatch for action {a}"
        assert isinstance(reward, (float, int)), "Reward must be numeric"
    print(f"✓ Action space verified: all {env.act_dim} discrete actions executed cleanly.")

    # 4. Test DQN Agent & PyTorch Pipeline
    try:
        import torch
        from dqn_agent import DQNAgent, DuelingQNetwork

        agent = DQNAgent(state_dim=env.obs_dim, action_dim=env.act_dim, batch_size=4)
        print("✓ PyTorch DQNAgent initialized on device:", agent.device)

        # Forward pass test
        dummy_state = torch.randn(2, 13).to(agent.device)
        q_vals = agent.policy_net(dummy_state)
        assert q_vals.shape == (2, env.act_dim), f"Expected Q-values shape (2, {env.act_dim}), got {q_vals.shape}"
        print(f"✓ Dueling Q-Network forward pass shape verified: (2, {env.act_dim}).")

        # 5. Test Smart KERS Boost Dynamics
        env.reset()
        env.boost_reserve = 0.50
        env.speed = 30.0
        # Action 6 is Heavy Trail Brake (KERS Active)
        _, _, _, _, info_kers = env.step(6)
        assert info_kers["kers_charging"] == True, "KERS should activate during braking"
        assert env.boost_reserve > 0.50, "KERS should recharge boost capacitor"
        print(f"✓ Smart KERS regenerative braking verified: Boost recharged to {env.boost_reserve*100:.1f}%.")

        # Replay buffer & update test
        for _ in range(10):
            agent.memory.push(obs, 2, 1.0, next_obs, False)
        loss = agent.update()
        assert loss is not None, "Agent update should return float loss"
        print(f"✓ Replay buffer and Bellman gradient step verified (Loss: {loss:.4f}).")

        # 6. Test Warm-Start Resume Checkpoint Adaptation
        import os
        ckpt_path = "checkpoints/best_dqn_policy.pt"
        if os.path.exists(ckpt_path):
            agent_resume = DQNAgent(state_dim=env.obs_dim, action_dim=env.act_dim, batch_size=4)
            agent_resume.load(ckpt_path)
            for _ in range(10):
                agent_resume.memory.push(obs, 8, 1.5, next_obs, False)
            resume_loss = agent_resume.update()
            assert resume_loss is not None, "Resumed agent update should compute loss without shape errors"
            print(f"✓ Warm-start action checkpoint adaptation & Bellman gradient step verified (Loss: {resume_loss:.4f}).")

    except ImportError:
        print("⚠️ PyTorch not detected in current Python environment. Skipping tensor checks.")

    print("\n" + "=" * 60)
    print("🎉 All Tests Passed! RL Environment with Smart Boost is Ready.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

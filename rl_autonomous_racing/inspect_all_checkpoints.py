import os
import torch
import numpy as np
from environment import ViceCoastRacingEnv
from dqn_agent import DQNAgent, DuelingQNetwork

checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") or f.endswith(".pth") or f.endswith(".ckpt")]

print(f"Found {len(files)} checkpoint file(s) in {checkpoint_dir}: {files}")

for fname in sorted(files):
    path = os.path.join(checkpoint_dir, fname)
    print("=" * 70)
    print(f"INSPECTING CHECKPOINT: {fname}")
    print(f"  File Size: {os.path.getsize(path):,} bytes")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            if "train_step_count" in data:
                print(f"  Train Step Count: {data['train_step_count']}")
            if "epsilon" in data:
                print(f"  Epsilon: {data['epsilon']}")
            
            p_net = data.get("policy_net", {})
            t_net = data.get("target_net", {})
            opt = data.get("optimizer", {})
            
            print(f"  Policy Net Weights: {len(p_net)} layers")
            for k, v in list(p_net.items()):
                print(f"    {k:35s}: shape {tuple(v.shape)}")
                
            in_features = p_net["feature_network.0.weight"].shape[1]
            hidden_features = p_net["feature_network.0.weight"].shape[0]
            val_out = p_net["value_stream.2.weight"].shape[0]
            adv_out = p_net["advantage_stream.2.weight"].shape[0]
            print(f"  Architecture: State Dim = {in_features}, Hidden = {hidden_features}, Value Stream Out = {val_out}, Advantage Out (Action Dim) = {adv_out}")
            
            # Check optimizer
            if opt:
                print(f"  Optimizer: {opt.get('param_groups', [{}])[0].get('lr', 'unknown')} lr")
        else:
            print(f"  Non-dict payload type: {type(data)}")
    except Exception as e:
        print(f"  Error loading: {e}")


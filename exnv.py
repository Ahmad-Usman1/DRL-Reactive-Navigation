import torch
import torch.nn as nn
from stable_baselines3 import PPO
import numpy as np
import onnxruntime as ort

# --- THE PURE PYTORCH WRAPPER ---
class PurePyTorchPolicy(nn.Module):
    """
    We completely sever all ties to Stable Baselines 3 methods.
    We extract the raw, native PyTorch layers (nn.Linear, nn.Flatten) 
    so the ONNX compiler only sees standard mathematical operations.
    """
    def __init__(self, sb3_policy):
        super().__init__()
        # 1. The flattener (usually nn.Flatten)
        self.features_extractor = sb3_policy.features_extractor
        
        # 2. The hidden neurons (nn.Sequential)
        self.policy_net = sb3_policy.mlp_extractor.policy_net
        
        # 3. The output neurons (nn.Linear)
        self.action_net = sb3_policy.action_net

    def forward(self, obs):
        # Pure tensor math. No SB3 extract_features() or predict() allowed here.
        x = self.features_extractor(obs)
        x = self.policy_net(x)
        action = self.action_net(x)
        return action


def export_and_verify():
    print("--- BEANS PURE PYTORCH EXPORTER & VERIFIER ---")
    
    # --- CONFIGURATION ---
    model_path = "Performing_Models/BEANS_Continued_v2_Final_6817600_steps"
    onnx_path = "Performing_Models/beans_verified.onnx"

    print(f"Loading PyTorch Master from: {model_path}")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        return

    # 1. ASSEMBLE THE PURE PYTORCH CLONE
    pure_policy = PurePyTorchPolicy(model.policy)
    pure_policy.eval()

    # 1 Batch, 27 Features
    dummy_input = torch.randn(1, 27, dtype=torch.float32)

    # 2. EXPORT
    print("\nExporting pure ONNX graph...")
    try:
        # FIX: The LeafSpec bug lives inside PyTorch's dynamo engine itself.
        # Every "modern" path (raw nn.Module, torch.export, torch.jit.trace)
        # eventually routes through it. dynamo=False forces the legacy
        # TorchScript exporter which has no dynamo involvement whatsoever.
        torch.onnx.export(
            pure_policy,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=["observation"],
            output_names=["action"],
            do_constant_folding=True,
            dynamo=False,           # Bypass dynamo entirely -- kills the LeafSpec bug
        )
        print(f"ONNX graph saved successfully to: {onnx_path}")
    except Exception as e:
        print(f"\n[FATAL EXPORT ERROR] PyTorch failed to compile the graph: {e}")
        return

    # 3. MATHEMATICAL VERIFICATION
    print("\n--- INITIATING MATHEMATICAL VERIFICATION ---")
    try:
        session = ort.InferenceSession(onnx_path)
    except Exception as e:
        print(f"FATAL ERROR loading ONNX Session: {e}")
        return

    for i in range(5):
        # Generate random raw observation data
        test_obs = np.random.uniform(-1.0, 1.0, size=(1, 27)).astype(np.float32)

        # PyTorch Inference (The Gold Standard)
        pt_action, _ = model.predict(test_obs[0], deterministic=True)

        # ONNX Inference (The Clone)
        ox_action = session.run(None, {"observation": test_obs})[0][0]

        # Calculate exact divergence
        diff = np.abs(pt_action - ox_action)
        max_diff = np.max(diff)

        print(f"\nTest {i+1}:")
        print(f"  PyTorch Action: {pt_action}")
        print(f"  ONNX Action:    {ox_action}")
        
        if max_diff < 1e-5:
            print("  Result: [PERFECT MATCH]")
        else:
            print(f"  Result: [FAILED] Divergence detected: {max_diff}")

    print("\nProtocol Complete. If you see [PERFECT MATCH], your pipeline is bulletproof.")


if __name__ == "__main__":
    export_and_verify()
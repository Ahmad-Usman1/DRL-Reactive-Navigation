import torch
import torch.nn as nn
from stable_baselines3 import PPO

class NakedPPOPolicy(nn.Module):
    """
    Corrected Naked Wrapper: 
    Preserves the feature extractor and the shared MLP layers while still 
    stripping away the non-deterministic probability distributions that crash ONNX.
    """
    def __init__(self, sb3_policy):
        super().__init__()
        # We store the entire policy object instead of breaking it apart
        self.policy = sb3_policy

    def forward(self, obs):
        # 1. Pass through the Flatten extractor
        features = self.policy.extract_features(obs)
        
        # 2. Pass through the full MLP (Shared Net -> Policy Net)
        # mlp_extractor returns a tuple: (latent_pi, latent_vf)
        # We only want the policy latent space (latent_pi)
        latent_pi, _ = self.policy.mlp_extractor(features)
        
        # 3. Output the raw deterministic action means
        return self.policy.action_net(latent_pi)
    """
    Override 4: Stripping Complex Outputs
    We strip away all Stable Baselines 3 metadata and environments, 
    exposing only the pure mathematical layers to ensure a raw tensor output.
    """
    def __init__(self, sb3_policy):
        super().__init__()
        self.policy_net = sb3_policy.mlp_extractor.policy_net
        self.action_net = sb3_policy.action_net

    def forward(self, obs):
        hidden = self.policy_net(obs)
        return self.action_net(hidden)

def export_beans_onnx_midas_method():
    print("--- BEANS ONNX Policy Extractor (The MiDaS Protocol) ---")
    model_path = "Performing_Models\\BEANS_FineTuned_3600000_steps.zip"
    onnx_path = "Performing_Models\\BEANS_FineTuned_3600000_steps.onnx"
    print(f"Loading heavy SB3 model from {model_path}...")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return
    
    # 1. Apply Naked Wrapper
    pure_policy = NakedPPOPolicy(model.policy)
    pure_policy.eval()

    # 27 is the Phase 3 observation space (17 rays + 4 kinematics + 6 action history)
    dummy_input = torch.randn(1, 27, dtype=torch.float32)

    # Overrides 1 & 2: JIT Trace with strict/check bypasses
    print("Tracing graph with strict=False and check_trace=False...")
    with torch.no_grad():
        traced_policy = torch.jit.trace(
            pure_policy, 
            dummy_input, 
            strict=False, 
            check_trace=False
        )

    # Override 3: Forcing the Legacy C++ Exporter
    print(f"Exporting to ONNX with dynamo=False to {onnx_path}...")
    torch.onnx.export(
        traced_policy,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=["observation"],
        output_names=["action"],
        dynamo=False # The hard override to kill the FX decomposition engine
    )
    print("SUCCESS! PPO agent exported using the MiDaS protocol.")

if __name__ == "__main__":
    export_beans_onnx_midas_method()
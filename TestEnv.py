import time
import numpy as np
from PeopleBotEnv import PeopleBotEnv 

def test_random_agent():
    print("--- Phase 1: Initializing Environment & MapBank ---")
    try:
        env = PeopleBotEnv()
    except Exception as e:
        print(f"FATAL ERROR on Init: {e}")
        return
    
    num_episodes = 5
    max_steps_per_ep = 500
    
    print("\n--- Phase 2: Stress Testing Physics & Raycaster ---")
    
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        total_reward = 0.0
        steps = 0
        
        start_time = time.time()
        
        while not (terminated or truncated) and steps < max_steps_per_ep:
            # 1. Sample entirely random motor commands
            action = env.action_space.sample()
            
            # 2. Step the physics engine
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # 3. Sanity check: Ensure no math broke
            if np.isnan(obs).any():
                print(f"\nFATAL ERROR: NaN detected in observation array at step {steps}!")
                print(f"Observation Dump: {obs}")
                return
                
        ep_time = time.time() - start_time
        fps = steps / ep_time if ep_time > 0 else 0
        
        reason = "Collision/Goal" if terminated else "Max Steps Reached"
        print(f"Episode {ep:02d} | Steps: {steps:03d} | Total Reward: {total_reward:7.2f} | FPS: {fps:5.0f} | Ended via: {reason}")

    print("\n--- TEST COMPLETE ---")
    print("Evaluate your results:")
    print("1. Did Episode 2+ hit thousands of FPS? (If yes, Numba is working).")
    print("2. Did it crash with a Gym OutOfBounds error? (If yes, our obs limits are wrong).")

if __name__ == "__main__":
    test_random_agent()
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from MapGenerator import MapGenerator

# --- CURRICULUM CONFIGURATION ---
SAVE_DIR = "training_maps"
# Notice the inclusion of the 0.10 transition tier
DIFFICULTIES = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0] 
MAPS_PER_TIER = 300
SIZE_X = 40
SIZE_Y = 40

def generate_single_map(args):
    """Worker function for parallel generation."""
    diff, index, save_path = args
    
    # Generate the map using your updated MapGenerator
    grid, wp, res = MapGenerator.generate(size_x=SIZE_X, size_y=SIZE_Y, difficulty=diff)
    
    # Save as compressed numpy array to save SSD space and RAM later
    filename = os.path.join(save_path, f"map_{index:04d}.npz")
    np.savez_compressed(filename, grid_map=grid, waypoints=wp, resolution=res)
    
    return True

def main():
    print("--- BEANS Curriculum Dataset Generator ---")
    
    # 1. Safety Check: Warn about dirty data
    if os.path.exists(SAVE_DIR) and len(os.listdir(SAVE_DIR)) > 0:
        print(f"WARNING: '{SAVE_DIR}' is not empty.")
        print("You MUST delete the old training maps before running this to prevent curriculum contamination.")
        resp = input("Type 'y' to proceed anyway, or 'n' to abort: ")
        if resp.lower() != 'y':
            print("Aborting. Go delete the folder and try again.")
            return

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. Build the Task List
    tasks = []
    for diff in DIFFICULTIES:
        tier_dir = os.path.join(SAVE_DIR, f"diff_{diff:.2f}")
        os.makedirs(tier_dir, exist_ok=True)
        for i in range(MAPS_PER_TIER):
            tasks.append((diff, i, tier_dir))

    total_maps = len(tasks)
    print(f"Curriculum Plan: {len(DIFFICULTIES)} Tiers x {MAPS_PER_TIER} Maps = {total_maps} Total Maps.")
    print("Engaging Multi-Core Processing (Your fans are about to spin up)...")

    # 3. Execute in Parallel
    # ProcessPoolExecutor automatically detects your CPU core count and maximizes throughput.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Wrap the executor in tqdm for a clean progress bar
        list(tqdm(executor.map(generate_single_map, tasks), total=total_maps, desc="Generating Maps"))

    print("\nDataset Generation Complete! MapBank is ready for training.")

if __name__ == "__main__":
    main()
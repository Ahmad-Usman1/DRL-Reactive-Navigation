import os
import random
import numpy as np

class MapBank:
    """
    Lazy-Loading Curriculum Bank.
    Keeps only file paths in RAM to save memory. 
    Loads the actual map data from disk only when requested.
    """
    def __init__(self, dataset_dir="training_maps"):
        self.dataset_dir = dataset_dir
        # Structure: { 0.0: [path1, path2...], 0.25: [path1, path2...], ... }
        self.binned_paths = {} 
        self.current_difficulty = 0.0
        
        self._index_dataset()

    def _index_dataset(self):
        """Builds an index of file paths without loading the data."""
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"CRITICAL: '{self.dataset_dir}' not found.")
            
        bin_folders = [f for f in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, f))]
        
        for folder in bin_folders:
            try:
                diff_val = float(folder.split('_')[1])
                self.binned_paths[diff_val] = []
                
                folder_path = os.path.join(self.dataset_dir, folder)
                # Store only the full path strings
                self.binned_paths[diff_val] = [
                    os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')
                ]
                
            except Exception as e:
                print(f"MapBank Warning: Skipping {folder}. Error: {e}")
        
        print("MapBank: Indexing Complete. (Memory footprint: ~1MB)")

    def set_difficulty(self, difficulty_level):
        self.current_difficulty = difficulty_level

    def get_random_map(self):
        """Loads a single map from disk on-the-fly."""
        closest_diff = min(self.binned_paths.keys(), key=lambda k: abs(k - self.current_difficulty))
        filepath = random.choice(self.binned_paths[closest_diff])
        
        # Load exactly what is needed, then the memory is freed after the reset()
        with np.load(filepath) as data:
            return data['grid_map'], data['waypoints'], data['resolution'].item()
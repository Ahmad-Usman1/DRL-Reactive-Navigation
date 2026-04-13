import numpy as np
import math

class ReactiveController:
    """
    Artificial Potential Field (APF) Controller for BEANS.
    A deterministic alternative to PPO.
    """
    def __init__(self, max_lin_vel=0.8, max_ang_vel=2.5):
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        
        # Gain constants - These are your "hyperparameters"
        self.k_att = 1.0  # Goal attraction strength
        self.k_rep = 0.5  # Obstacle repulsion strength
        self.safe_dist = 0.25 # Normalized (0.25 * 5m = 1.25m)

    def predict(self, obs):
        """
        Takes the 24-dim observation and returns [lin_action, ang_action].
        Matches the SB3 model.predict() interface.
        """
        # 1. Unpack the Augmented Observation
        # obs: [20 lidar, 1 dist, 1 head, 1 v, 1 w]
        scan = obs[0:20]
        goal_dist = obs[20] # Normalized 0-1 (0 to 10m)
        goal_head = obs[21] # Normalized -1 to 1 (-pi to pi)
        curr_v = obs[22]
        curr_w = obs[23]

        # 2. Calculate Attractive Force (Goal)
        # We want to point toward goal_head.
        f_att_x = math.cos(goal_head * np.pi)
        f_att_y = math.sin(goal_head * np.pi)

        # 3. Calculate Repulsive Force (Obstacles)
        f_rep_x = 0.0
        f_rep_y = 0.0
        
        # Sensor angles match your PeopleBotEnv setup
        front_degs = [90, 50, 30, 20, 10, 5, 0, -5, -10, -20, -30, -50, -90]
        rear_degs = [130, 150, 170, 180, -170, -150, -130]
        angles = np.deg2rad(front_degs + rear_degs)

        for i in range(20):
            dist = scan[i] # Normalized 0.0 to 1.0
            if dist < self.safe_dist:
                # Repulsion is inversely proportional to distance
                # Use a small epsilon to avoid division by zero
                strength = self.k_rep * (1.0 / (dist + 0.05) - 1.0 / self.safe_dist)
                # Force points AWAY from the sensor ray
                f_rep_x -= strength * math.cos(angles[i])
                f_rep_y -= strength * math.sin(angles[i])

        # 4. Resultant Vector
        total_x = self.k_att * f_att_x + f_rep_x
        total_y = self.k_att * f_att_y + f_rep_y
        
        desired_heading = math.atan2(total_y, total_x)
        magnitude = math.sqrt(total_x**2 + total_y**2)

        # 5. Map to Actions [-1, 1]
        # Linear velocity: Slow down if we need to turn sharply
        lin_action = math.cos(desired_heading) * min(1.0, magnitude)
        
        # Angular velocity: Proportional to heading error
        ang_action = desired_heading / np.pi # Scale to [-1, 1]

        # 6. Safety Override: If something is dead ahead, FORCE a stop/turn
        min_front = np.min(scan[4:9]) # Check indices 4-8 (the central cone)
        if min_front < 0.15: # Roughly 0.75m
            lin_action = -0.2 # Back up slightly
            ang_action = 1.0 if goal_head > 0 else -1.0 # Hard turn

        return [np.clip(lin_action, -1, 1), np.clip(ang_action, -1, 1)], None
import numpy as np
import math

class DWAConfig:
    def __init__(self):
        # Kinematics (Matches PeopleBotEnv exactly)
        self.max_speed = 0.4
        self.min_speed = 0.0
        self.max_yaw_rate = 1.9
        
        # Accelerations (Estimated based on tau_motor = 0.3)
        self.max_accel = 1.5       # m/s^2
        self.max_delta_yaw_rate = 4.0 # rad/s^2
        
        # Resolutions for the search grid
        self.v_resolution = 0.05
        self.yaw_rate_resolution = 0.1
        
        self.dt = 0.1
        self.predict_time = 2.0
        self.robot_radius = 0.31

def dwa_control(x, config, goal, ob, alpha, beta, gamma):
    """
    x: [x, y, theta, v, w]
    ob: list of [x, y] obstacle points from the 17-ray scan
    Returns: [v, w], best_trajectory
    """
    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob, alpha, beta, gamma)
    return u, trajectory

def calc_dynamic_window(x, config):
    # Vs: Robot kinematic limits
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Vd: Dynamic limits based on current velocity and max acceleration
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    # Window is the intersection of Vs and Vd
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

def predict_trajectory(x_init, v, y, config):
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = kinematic_model(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory

def kinematic_model(x, u, dt):
    # x = [x, y, theta, v, w]
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_clearance(trajectory, ob, config):
    if len(ob) == 0:
        return float('inf')
        
    clearance = float('inf')
    # Check all points in the trajectory against all obstacles
    for i in range(len(trajectory)):
        rx = trajectory[i, 0]
        ry = trajectory[i, 1]
        
        # Vectorized distance calculation
        dx = ob[:, 0] - rx
        dy = ob[:, 1] - ry
        r = np.hypot(dx, dy)
        min_r = np.min(r)
        
        if min_r < config.robot_radius:
            return -float('inf') # Collision
            
        if min_r < clearance:
            clearance = min_r
            
    return clearance

def calc_heading(trajectory, goal):
    # The heading reward is based on the angle difference between the 
    # final trajectory position and the goal.
    last_x = trajectory[-1, 0]
    last_y = trajectory[-1, 1]
    last_theta = trajectory[-1, 2]
    
    desired_theta = math.atan2(goal[1] - last_y, goal[0] - last_x)
    error = (desired_theta - last_theta + math.pi) % (2 * math.pi) - math.pi
    
    # Cost is the absolute error. Reward is pi - cost.
    return math.pi - abs(error)

def calc_control_and_trajectory(x, dw, config, goal, ob, alpha, beta, gamma):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    
    ob_np = np.array(ob) if len(ob) > 0 else np.empty((0, 2))

    # To properly tune DWA, we must normalize the objectives. 
    # We store them first, find maxes, then evaluate.
    trajectories = []
    
    for v in np.arange(dw[0], dw[1] + config.v_resolution, config.v_resolution):
        for y in np.arange(dw[2], dw[3] + config.yaw_rate_resolution, config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)
            
            clearance = calc_clearance(trajectory, ob_np, config)
            if clearance == -float('inf'):
                continue # Discard collisions
                
            heading = calc_heading(trajectory, goal)
            velocity = v
            
            trajectories.append({
                "u": [v, y],
                "traj": trajectory,
                "h": heading,
                "c": clearance,
                "v": velocity
            })
            
    if not trajectories:
        # Recovery behavior: Spin in place
        return [0.0, -config.max_yaw_rate], np.array([x])

    # Normalization
    max_h = max([t["h"] for t in trajectories]) or 1.0
    max_c = max([t["c"] for t in trajectories]) or 1.0
    max_v = max([t["v"] for t in trajectories]) or 1.0

    best_score = -float('inf')
    
    for t in trajectories:
        # Normalized Score
        score = (alpha * (t["h"]/max_h)) + \
                (beta * (t["c"]/max_c)) + \
                (gamma * (t["v"]/max_v))
                
        if score > best_score:
            best_score = score
            best_u = t["u"]
            best_trajectory = t["traj"]

    return best_u, best_trajectory
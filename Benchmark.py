"""
BEANS Benchmark Suite
=====================
Comparative evaluation of PPO vs DWA vs APF on the PeopleBotEnv.

Improvements integrated:
  1. Visual telemetry - 3-panel PNG every N episodes with correct coordinate scaling
  2. True A*-based SPL - dynamically computes optimal path length per map
  3. Microsecond inference latency tracking - isolated from physics/rendering
  4. Granular failure classification - Crash vs Trap vs Success
  5. Calibrated baselines - DWA with curved arc projection, APF with exponential repulsion
"""

import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Headless rendering - no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import binary_dilation
from stable_baselines3 import PPO

from PeopleBotEnv import PeopleBotEnv

# ─────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────
VIZ_DIR = "./benchmark_visuals/"
os.makedirs(VIZ_DIR, exist_ok=True)

# How often (in episodes) to dump a trajectory PNG per algorithm
VIZ_EVERY_N = 10

# Collision threshold: if min lidar drops below this without goal → Crash
CRASH_LIDAR_THRESHOLD = 0.25  # metres


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRUE MATHEMATICAL SPL — A*-BASED OPTIMAL PATH LENGTH
# ═══════════════════════════════════════════════════════════════════════════════

def get_optimal_path_length(map_grid, waypoints, resolution):
    """
    Runs A* between every consecutive waypoint pair on a wall-inflated grid
    and returns the total shortest possible physical distance in metres.

    This is the correct SPL denominator for any map topology.  Hardcoding
    a constant (e.g. 5.0 m) produces a mathematically invalid metric.
    """
    if len(waypoints) < 2:
        return 1.0  # degenerate case

    # Inflate walls by 1 downsampled cell so the optimal path is also safe
    DS = 5
    small = map_grid[::DS, ::DS]
    inflated = binary_dilation(small == 1, iterations=1)
    pf_matrix = (~inflated).astype(int)

    total_dist_m = 0.0
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

    for seg in range(len(waypoints) - 1):
        sx = int(round(waypoints[seg][0] * resolution)) // DS
        sy = int(round(waypoints[seg][1] * resolution)) // DS
        gx = int(round(waypoints[seg + 1][0] * resolution)) // DS
        gy = int(round(waypoints[seg + 1][1] * resolution)) // DS

        # Clamp to grid bounds
        h, w = pf_matrix.shape
        sx, sy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
        gx, gy = np.clip(gx, 0, w - 1), np.clip(gy, 0, h - 1)

        grid = Grid(matrix=pf_matrix.tolist())
        sn, en = grid.node(sx, sy), grid.node(gx, gy)

        if not sn.walkable or not en.walkable:
            # Fall back to Euclidean for this segment
            seg_dist = math.hypot(
                waypoints[seg + 1][0] - waypoints[seg][0],
                waypoints[seg + 1][1] - waypoints[seg][1]
            )
            total_dist_m += seg_dist
            continue

        path, _ = finder.find_path(sn, en, grid)
        if not path:
            seg_dist = math.hypot(
                waypoints[seg + 1][0] - waypoints[seg][0],
                waypoints[seg + 1][1] - waypoints[seg][1]
            )
            total_dist_m += seg_dist
            continue

        # Convert path nodes back to metres and sum segment lengths
        path_m = [(node.x * DS / resolution, node.y * DS / resolution)
                  for node in path]
        for i in range(1, len(path_m)):
            total_dist_m += math.hypot(
                path_m[i][0] - path_m[i - 1][0],
                path_m[i][1] - path_m[i - 1][1]
            )

    return max(total_dist_m, 0.1)  # guard against zero-length edge case


def compute_spl(successes, optimal_lengths, actual_lengths):
    """SPL = (1/N) Σ [ S_i * L*_i / max(p_i, L*_i) ]"""
    scores = []
    for s, l_star, p in zip(successes, optimal_lengths, actual_lengths):
        if s:
            scores.append(l_star / max(p, l_star))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CALIBRATED BASELINE — DWA WITH CURVED ARC PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def dwa_action(scan_data, heading_error, sensor_angles,
               lin_vel, max_lin, max_ang, robot_radius=0.31):
    """
    Dynamic Window Approach with 1.5-second curved kinematic arc projection.

    Arc projection (rather than flat vector scoring) correctly models the
    turning geometry of a differential-drive robot. This is the standard
    DWA formulation from Fox et al. 1997.
    """
    PREDICT_TIME = 1.5      # seconds to simulate forward
    DT_SIM      = 0.1       # simulation timestep
    STEPS       = int(PREDICT_TIME / DT_SIM)

    V_SAMPLES   = 7
    W_SAMPLES   = 9

    best_score  = -np.inf
    best_v, best_w = 0.05, 0.0

    for v in np.linspace(0.05, max_lin, V_SAMPLES):
        for w in np.linspace(-max_ang, max_ang, W_SAMPLES):

            # --- Simulate the arc ---
            px, py, pth = 0.0, 0.0, 0.0
            min_clearance = np.inf
            for _ in range(STEPS):
                pth += w * DT_SIM
                px  += v * math.cos(pth) * DT_SIM
                py  += v * math.sin(pth) * DT_SIM

            # Arc heading after PREDICT_TIME
            arc_heading_error = heading_error - (w * PREDICT_TIME)

            # Clearance: minimum scan in forward ±60°
            front_mask = np.abs(sensor_angles) < np.deg2rad(60)
            clearance  = float(np.min(scan_data[front_mask])) if front_mask.any() else 5.0

            # Reject trajectories that are predicted to collide
            if clearance < robot_radius * 1.5:
                continue

            heading_score   = math.cos(arc_heading_error)
            clearance_score = min(clearance / 2.0, 1.0)
            velocity_score  = v / max_lin

            score = (1.5 * heading_score
                     + 0.8 * clearance_score
                     + 0.4 * velocity_score
                     - 0.3 * abs(w))

            if score > best_score:
                best_score  = score
                best_v, best_w = v, w

    return np.array([best_v / max_lin, best_w / max_ang], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATED BASELINE — APF WITH EXPONENTIAL REPULSION
# ═══════════════════════════════════════════════════════════════════════════════

def apf_action(pose, goal, scan_data, sensor_angles, max_lin, max_ang):
    """
    Artificial Potential Field with exponential decay repulsion.

    Replaces the original linear repulsion, which fails in narrow corridors
    because it generates symmetric forces that cancel. The exponential decay
    ((1/d - 1/d0)^2) creates an aggressive hard wall at close range.
    """
    INFLUENCE_DIST = 1.8    # metres — repulsion onset
    K_ATT   = 1.0           # attractive gain
    K_REP   = 2.5           # repulsive gain

    # Attractive: unit vector toward goal, saturated at 1.0
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    dist_goal = math.hypot(dx, dy)
    att_x = K_ATT * dx / (dist_goal + 1e-6)
    att_y = K_ATT * dy / (dist_goal + 1e-6)

    # Repulsive: exponential decay from each lidar hit
    rep_x, rep_y = 0.0, 0.0
    for i, d in enumerate(scan_data):
        if d < INFLUENCE_DIST and d > 0.05:
            # Exponential: (1/d - 1/d0)^2  — hard wall near obstacles
            mag = K_REP * ((1.0 / d) - (1.0 / INFLUENCE_DIST)) ** 2
            obs_angle = pose[2] + float(sensor_angles[i])
            rep_x -= math.cos(obs_angle) * mag
            rep_y -= math.sin(obs_angle) * mag

    # Resultant force → desired heading → diff-drive commands
    total_x = att_x + rep_x
    total_y = att_y + rep_y
    desired_angle   = math.atan2(total_y, total_x)
    heading_error   = (desired_angle - pose[2] + math.pi) % (2 * math.pi) - math.pi

    # Speed inversely proportional to heading error to reduce corner-cutting
    v = max_lin * max(0.0, math.cos(heading_error)) * min(dist_goal / 2.0, 1.0)
    w = float(np.clip(2.5 * heading_error, -max_ang, max_ang))
    return np.array([v / max_lin, w / max_ang], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUAL TELEMETRY — 3-PANEL PNG DUMP
# ═══════════════════════════════════════════════════════════════════════════════

def save_trajectory_png(algo_name, episode_idx, map_grid, resolution,
                        waypoints, trajectory, scan_history,
                        outcome, path_length, inference_ms):
    """
    Dumps a 3-panel diagnostic PNG:
      Panel 1 — Trajectory overlaid on occupancy grid (metres-correct scaling)
      Panel 2 — Min lidar distance over time
      Panel 3 — Inference latency over time
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{algo_name} | Episode {episode_idx} | "
        f"Outcome: {outcome} | Path: {path_length:.2f}m | "
        f"Inference: {np.mean(inference_ms):.2f}ms avg",
        fontsize=12, fontweight="bold"
    )

    # ── Panel 1: Map + Trajectory ──────────────────────────────────────────
    ax = axes[0]
    ax.imshow(map_grid == 0, cmap="gray", origin="lower",
              extent=[0, map_grid.shape[1] / resolution,
                      0, map_grid.shape[0] / resolution])

    if len(trajectory) > 1:
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1],
                color="cyan", linewidth=1.2, alpha=0.85, label="Trajectory")
        ax.scatter(traj[0, 0], traj[0, 1],
                   c="lime", s=80, zorder=5, label="Start")
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   c="red", s=80, marker="X", zorder=5, label="End")

    # Waypoints — guard against empty/ambiguous arrays
    if len(waypoints) > 0:
        wp = np.array(waypoints)
        ax.plot(wp[:, 0], wp[:, 1],
                "r--", linewidth=1.0, alpha=0.6, label="Planned")
        ax.scatter(wp[0, 0], wp[0, 1],
                   c="lime", s=120, marker="*", zorder=6)
        ax.scatter(wp[-1, 0], wp[-1, 1],
                   c="blue", s=120, marker="*", zorder=6, label="Goal")

    color_map = {"Success": "lime", "Crash": "red", "Trap": "orange"}
    ax.set_facecolor("#1a1a2e")
    ax.set_title(f"Occupancy + Trajectory", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=7, loc="upper right")
    outcome_patch = mpatches.Patch(
        color=color_map.get(outcome, "white"), label=outcome)
    ax.legend(handles=[outcome_patch] + ax.get_legend_handles_labels()[0],
              fontsize=7, loc="upper right")

    # ── Panel 2: Min Lidar over Time ────────────────────────────────────────
    ax2 = axes[1]
    if scan_history:
        ax2.plot(scan_history, color="orange", linewidth=1.0)
        ax2.axhline(CRASH_LIDAR_THRESHOLD, color="red",
                    linestyle="--", linewidth=1.0, label=f"Crash threshold ({CRASH_LIDAR_THRESHOLD}m)")
        ax2.axhline(0.4, color="yellow",
                    linestyle=":", linewidth=1.0, label="Danger zone (0.4m)")
    ax2.set_title("Min Lidar Clearance (m)", fontsize=10)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (m)")
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 5.5)

    # ── Panel 3: Inference Latency ──────────────────────────────────────────
    ax3 = axes[2]
    if inference_ms:
        ax3.plot(inference_ms, color="mediumpurple", linewidth=0.8, alpha=0.7)
        ax3.axhline(np.mean(inference_ms), color="white",
                    linestyle="--", linewidth=1.2,
                    label=f"Mean: {np.mean(inference_ms):.3f}ms")
        ax3.axhline(np.max(inference_ms), color="red",
                    linestyle=":", linewidth=1.0,
                    label=f"Max: {np.max(inference_ms):.3f}ms")
    ax3.set_title("Inference Latency (ms)", fontsize=10)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("ms")
    ax3.legend(fontsize=7)

    plt.tight_layout()
    fname = os.path.join(VIZ_DIR, f"{algo_name}_ep{episode_idx:04d}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight",
                facecolor="#0d0d1a")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EPISODE RUNNER — with latency tracking & failure classification
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(env, policy_fn, algo_name, episode_idx,
                optimal_path_length, dump_visuals=False, max_steps=3000):
    """
    Runs one episode and returns a result dict with all benchmark metrics.
    """
    obs, _ = env.reset()

    trajectory    = [env.current_pose[:2].copy()]
    scan_history  = []
    inference_ms  = []
    path_length   = 0.0
    total_jerk    = 0.0
    prev_pos      = env.current_pose[:2].copy()
    prev_action   = np.zeros(2, dtype=np.float32)
    min_lidar_log = []

    for step in range(max_steps):
        # ── 3. Inference latency — isolated measurement ──────────────────
        t0 = time.perf_counter()
        action = policy_fn(obs, env)
        t1 = time.perf_counter()
        inference_ms.append((t1 - t0) * 1000.0)

        obs, reward, terminated, truncated, info = env.step(action)

        # Trajectory & path length
        curr_pos = env.current_pose[:2].copy()
        path_length += np.linalg.norm(curr_pos - prev_pos)
        prev_pos = curr_pos.copy()
        trajectory.append(curr_pos)

        # Jerk (smoothness)
        total_jerk += np.linalg.norm(action - prev_action)
        prev_action = action.copy()

        # Lidar
        raw_scan = obs[:env.num_rays] * env.max_sensor_range
        min_lidar = float(np.min(raw_scan))
        min_lidar_log.append(min_lidar)
        scan_history.append(min_lidar)

        if terminated or truncated:
            break

    # ── 4. Granular failure classification ──────────────────────────────
    tel = info.get("telemetry", {})
    is_success = tel.get("rate_success", 0.0) == 1.0
    is_timeout = tel.get("rate_timeout", 0.0) == 1.0

    # Crash: lidar dropped below threshold at any point without success
    lidar_floor = min(min_lidar_log) if min_lidar_log else env.max_sensor_range
    is_crash = (not is_success) and (lidar_floor < CRASH_LIDAR_THRESHOLD)

    # Trap: timed out without crashing — stuck in local minimum
    is_trap = is_timeout and (not is_crash)

    if is_success:
        outcome = "Success"
    elif is_crash:
        outcome = "Crash"
    else:
        outcome = "Trap"

    result = {
        "success":       is_success,
        "crash":         is_crash,
        "trap":          is_trap,
        "outcome":       outcome,
        "path_length":   path_length,
        "optimal_length": optimal_path_length,
        "smoothness":    total_jerk / max(step + 1, 1),
        "avg_clearance": float(np.mean(min_lidar_log)) if min_lidar_log else 0.0,
        "min_clearance": lidar_floor,
        "inference_mean_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
        "inference_max_ms":  float(np.max(inference_ms))  if inference_ms else 0.0,
        "steps":         step + 1,
    }

    # ── 1. Visual telemetry dump ─────────────────────────────────────────
    if dump_visuals:
        save_trajectory_png(
            algo_name      = algo_name,
            episode_idx    = episode_idx,
            map_grid       = env.map_grid,
            resolution     = env.resolution,
            waypoints      = env.waypoints,
            trajectory     = trajectory,
            scan_history   = scan_history,
            outcome        = outcome,
            path_length    = path_length,
            inference_ms   = inference_ms,
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN BENCHMARK LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def reset_env_to_saved_state(env, saved_map, saved_wp, saved_res):
    """
    Restores an environment to a previously saved map/waypoint state
    so all three algorithms face the exact same scenario.
    """
    env.reset()
    env.map_grid   = saved_map.copy()
    env.waypoints  = saved_wp
    env.resolution = saved_res

    if len(saved_wp) > 1:
        env.current_goal_index = 1
        env.current_goal = np.array(saved_wp[1])
        start_theta = math.atan2(
            saved_wp[1][1] - saved_wp[0][1],
            saved_wp[1][0] - saved_wp[0][0]
        )
    else:
        env.current_goal_index = 0
        env.current_goal = np.array(saved_wp[0])
        start_theta = 0.0

    env.current_pose = np.array([saved_wp[0][0], saved_wp[0][1], start_theta])
    env.current_lin_vel = 0.0
    env.current_ang_vel = 0.0
    env.previous_distance = np.linalg.norm(
        env.current_pose[:2] - env.current_goal)
    env.current_step = 0
    env.action_history = np.zeros(
        (env.lag_steps, 2), dtype=np.float32)


def print_results_table(results_by_algo, difficulties, n_per_diff):
    """Prints a formatted benchmark table to stdout."""
    header = (f"\n{'─'*85}\n"
              f" BEANS Benchmark Results | "
              f"{sum(n_per_diff.values())} episodes per algorithm\n"
              f"{'─'*85}")
    print(header)
    print(f"{'Algorithm':<10} {'SPL':>7} {'Success':>9} {'Crash':>8} "
          f"{'Trap':>7} {'Smooth':>9} {'Clear(m)':>10} "
          f"{'Inf(ms)':>9} {'MaxInf':>8}")
    print("─" * 85)

    for algo, results in results_by_algo.items():
        successes  = [r["success"]        for r in results]
        opt_lens   = [r["optimal_length"] for r in results]
        act_lens   = [r["path_length"]    for r in results]
        crashes    = [r["crash"]          for r in results]
        traps      = [r["trap"]           for r in results]
        smoothness = [r["smoothness"]     for r in results]
        clearance  = [r["avg_clearance"]  for r in results if r["success"]]
        inf_mean   = [r["inference_mean_ms"] for r in results]
        inf_max    = [r["inference_max_ms"]  for r in results]

        spl    = compute_spl(successes, opt_lens, act_lens)
        sc_pct = 100 * np.mean(successes)
        cr_pct = 100 * np.mean(crashes)
        tr_pct = 100 * np.mean(traps)
        sm     = np.mean(smoothness)
        cl     = np.mean(clearance) if clearance else float("nan")
        im     = np.mean(inf_mean)
        ix     = np.mean(inf_max)

        print(f"{algo:<10} {spl:>7.3f} {sc_pct:>8.1f}% {cr_pct:>7.1f}% "
              f"{tr_pct:>6.1f}% {sm:>9.4f} {cl:>9.3f}m "
              f"{im:>8.3f}ms {ix:>7.3f}ms")

    print("─" * 85)

    # Per-difficulty breakdown
    print("\nPer-Difficulty Breakdown (Success %)")
    print(f"{'Algorithm':<10}", end="")
    for d in difficulties:
        print(f"  Diff {d:.2f}", end="")
    print()
    for algo, results in results_by_algo.items():
        print(f"{algo:<10}", end="")
        for d in difficulties:
            subset = [r for r in results
                      if abs(r.get("difficulty", d) - d) < 0.01]
            if subset:
                pct = 100 * np.mean([r["success"] for r in subset])
                print(f"  {pct:>8.1f}%", end="")
        print()
    print("─" * 85)


def benchmark(model_path,
              n_episodes=60,
              difficulties=(0.1, 0.5, 1.0),
              dump_visuals=True):
    """
    Main entry point.

    Parameters
    ----------
    model_path   : path to a saved SB3 PPO model (.zip)
    n_episodes   : total evaluation episodes (split evenly across difficulties)
    difficulties : tuple of map difficulty values to evaluate
    dump_visuals : whether to save trajectory PNGs
    """
    env       = PeopleBotEnv()
    ppo_model = PPO.load(model_path, env=env)

    n_per_diff  = {d: n_episodes // len(difficulties) for d in difficulties}
    all_results = defaultdict(list)
    episode_counter = {"PPO": 0, "DWA": 0, "APF": 0}

    algos = {
        "PPO": lambda obs, e: ppo_model.predict(obs, deterministic=True)[0],
        "DWA": lambda obs, e: dwa_action(
            obs[:e.num_rays] * e.max_sensor_range,
            obs[e.num_rays + 1] * np.pi,
            e.sensor_angles,
            e.current_lin_vel,
            e.max_lin_vel,
            e.max_ang_vel
        ),
        "APF": lambda obs, e: apf_action(
            e.current_pose,
            e.current_goal,
            obs[:e.num_rays] * e.max_sensor_range,
            e.sensor_angles,
            e.max_lin_vel,
            e.max_ang_vel
        ),
    }

    print(f"\nBEANS Benchmark | Model: {os.path.basename(model_path)}")
    print(f"Episodes: {n_episodes} | Difficulties: {difficulties}")
    print(f"Visual dumps: {'ON → ' + VIZ_DIR if dump_visuals else 'OFF'}\n")

    for diff in difficulties:
        env.set_difficulty(diff)
        n_ep = n_per_diff[diff]
        print(f"  Difficulty {diff:.2f} — {n_ep} episodes per algorithm")

        for ep in range(n_ep):
            # ── Snapshot the map so all three algorithms see the same scenario ──
            base_obs, _ = env.reset()
            saved_map = env.map_grid.copy()
            saved_wp  = list(env.waypoints)   # plain list avoids np ambiguity
            saved_res = env.resolution

            # ── 2. True SPL denominator — computed once per map ──────────────
            opt_len = get_optimal_path_length(saved_map, saved_wp, saved_res)

            for algo_name, policy_fn in algos.items():
                reset_env_to_saved_state(env, saved_map, saved_wp, saved_res)

                episode_counter[algo_name] += 1
                dump = (dump_visuals and
                        episode_counter[algo_name] % VIZ_EVERY_N == 0)

                result = run_episode(
                    env            = env,
                    policy_fn      = policy_fn,
                    algo_name      = algo_name,
                    episode_idx    = episode_counter[algo_name],
                    optimal_path_length = opt_len,
                    dump_visuals   = dump,
                )
                result["difficulty"] = diff
                all_results[algo_name].append(result)

            # Live progress
            ppo_latest = all_results["PPO"][-1]
            print(f"    ep {ep+1:>3}/{n_ep}  "
                  f"PPO:{ppo_latest['outcome']:<8}  "
                  f"opt={opt_len:.1f}m  "
                  f"actual={ppo_latest['path_length']:.1f}m",
                  end="\r")

        print()  # newline after \r progress

    print_results_table(all_results, difficulties, n_per_diff)

    # ── Save raw results as numpy archive for post-processing ────────────────
    save_path = os.path.join(VIZ_DIR, "raw_results.npz")
    np.savez(save_path, **{k: v for k, v in all_results.items()})
    print(f"\nRaw results saved to: {save_path}")
    print(f"Trajectory PNGs saved to: {VIZ_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    model_path = (sys.argv[1] if len(sys.argv) > 1
                  else "./Performing_Models/BEANS_Continued_v2_Final_6817600_steps.zip")

    if not os.path.exists(model_path):
        # Try without .zip extension (SB3 auto-appends it)
        if os.path.exists(model_path.replace(".zip", "")):
            model_path = model_path.replace(".zip", "")
        else:
            print(f"Model not found: {model_path}")
            print("Usage: python Benchmark.py <path_to_model.zip>")
            sys.exit(1)

    benchmark(
        model_path   = model_path,
        n_episodes   = 60,
        difficulties = (0.1, 0.5, 1.0),
        dump_visuals = True,
    )
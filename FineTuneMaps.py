"""
FineTuneMaps.py
===============
Specialized map generators for the corner/narrow-corridor finetuning phase.

Map types and their targeted failure modes:
  1. CornerGauntlet  — Tests velocity reduction at 90° turns (the primary crash cause)
  2. PinchPoint      — Tests commitment through a 1.5–1.8m neck (vs. detour preference)
  3. ForkTrap        — Tests correct branch selection at a T-junction (anti-local-minima)
  4. ClutterCorridor — Tests general close-quarters navigation and weaving

All maps guarantee a minimum 1.5m clearance in traversable corridors, consistent
with the robot's physical footprint (radius=0.31m) and the 0.45m safety inflation
used in MapGenerator's A* planning.

FineTuneMapBank mixes these (70%) with standard-difficulty maps (30%) to prevent
catastrophic forgetting of the base navigation policy.
"""

import numpy as np
import random
import math

# Must match PeopleBotEnv.resolution = 50
RESOLUTION = 50  # pixels per meter

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _carve(grid, y_lo, y_hi, x_lo, x_hi):
    """Safely carve a free-space rectangle into the grid."""
    h, w = grid.shape
    y_lo = max(0, y_lo)
    y_hi = min(h, y_hi)
    x_lo = max(0, x_lo)
    x_hi = min(w, x_hi)
    if y_hi > y_lo and x_hi > x_lo:
        grid[y_lo:y_hi, x_lo:x_hi] = 0


def _seal_border(grid, border_px=5):
    """Enforce solid border walls so the robot can never leave the map."""
    grid[:border_px, :] = 1
    grid[-border_px:, :] = 1
    grid[:, :border_px] = 1
    grid[:, -border_px:] = 1


# ---------------------------------------------------------------------------
# Map 1: Corner Gauntlet
# ---------------------------------------------------------------------------

class CornerGauntletMap:
    """
    A zigzag path of 2–4 right-angle turns connected by 2m-wide corridors.

    Targeted failure mode: the robot enters a turn at full speed because
    front_dist is large, but the wall closes in from the side as it turns.
    The corner-aware velocity budget in FineTuneBotEnv is designed to fix this.

    Generation strategy:
      - Place turning points that alternate between top/bottom thirds of the map.
      - For each consecutive pair, carve an L-shape: first horizontal at the source
        y-level, then vertical to reach the target y-level. The corner junction is
        automatically included in both carve operations.
    """

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res)
        h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corr_px = int(2.0 * res)   # 2m corridor width
        half    = corr_px // 2

        n_turns = random.randint(2, 4)
        margin  = int(3.5 * res)   # keep start/end away from border

        # Build the centerline: start → n_turns intermediate → end
        pts = [(margin, h // 2)]
        x_step = (w - 2 * margin) // (n_turns + 1)

        for i in range(n_turns):
            x = margin + (i + 1) * x_step
            if i % 2 == 0:
                y = random.randint(int(0.22 * h), int(0.38 * h))
            else:
                y = random.randint(int(0.62 * h), int(0.78 * h))
            pts.append((x, y))

        pts.append((w - margin, h // 2))

        # Carve each L-shaped segment
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]

            # Horizontal leg at source height y0
            x_lo = min(x0, x1) - half
            x_hi = max(x0, x1) + half
            _carve(grid, y0 - half, y0 + half, x_lo, x_hi)

            # Vertical leg at destination x1
            y_lo = min(y0, y1) - half
            y_hi = max(y0, y1) + half
            _carve(grid, y_lo, y_hi, x1 - half, x1 + half)

        _seal_border(grid)

        # Waypoints: one per turning point, in metres, clamped to safe zone
        waypoints_list = []
        for px, py in pts:
            wx = float(np.clip(px / res, 2.0, size_x - 2.0))
            wy = float(np.clip(py / res, 2.0, size_y - 2.0))
            waypoints_list.append([wx, wy])

        waypoints = np.array(waypoints_list, dtype=np.float32)

        # ── RANDOMIZATION: Geometric Matrix Transformations ──
        # Flip Horizontally (Makes it right-to-left)
        if random.choice([True, False]): 
            grid = np.fliplr(grid)
            waypoints[:, 0] = size_x - waypoints[:, 0]
            
        # Flip Vertically (Inverts the top/bottom turns)
        if random.choice([True, False]): 
            grid = np.flipud(grid)
            waypoints[:, 1] = size_y - waypoints[:, 1]

        # Transpose Map (Changes East/West progression to North/South progression)
        if random.choice([True, False]):
            grid = grid.T
            # Swap X and Y coordinates in the waypoints array
            waypoints[:, [0, 1]] = waypoints[:, [1, 0]]

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 2: Pinch Point
# ---------------------------------------------------------------------------

class PinchPointMap:
    """
    Wide start room ──> 1.5–1.8m neck ──> wide goal room.

    Targeted failure mode: the robot detects the narrow passage and prefers
    to orbit the start room rather than commit. The narrow-passage bonus in
    FineTuneBotEnv rewards continued forward progress through the neck.

    The neck width is randomised between 1.5m (absolute minimum) and 1.8m
    to provide variety while always staying above the robot's physical width.
    """

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res)
        h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        center_y  = h // 2
        # Neck geometry
        neck_w_m  = random.uniform(1.5, 1.8)
        neck_w_px = int(neck_w_m * res)
        neck_len  = int(random.uniform(5.0, 9.0) * res)
        # Room geometry
        room_sz   = int(random.uniform(6.0, 9.0) * res)

        # Horizontal centre of the neck
        neck_x = w // 2 - neck_len // 2
        neck_y = center_y - neck_w_px // 2

        # --- Left room ---
        lr_x = max(int(0.05 * w), neck_x - room_sz)
        lr_y = center_y - room_sz // 2
        _carve(grid, lr_y, lr_y + room_sz, lr_x, neck_x + neck_w_px)

        # --- Neck ---
        _carve(grid, neck_y, neck_y + neck_w_px, neck_x, neck_x + neck_len)

        # --- Right room ---
        rr_x = neck_x + neck_len
        rr_y = center_y - room_sz // 2
        _carve(grid, rr_y, rr_y + room_sz, rr_x, min(int(0.95 * w), rr_x + room_sz))

        _seal_border(grid)

        # Waypoints: start → neck entry → neck mid → neck exit → goal
        start_x  = float(np.clip((lr_x + room_sz * 0.3) / res, 2.0, size_x - 2.0))
        entry_x  = float(np.clip((neck_x + neck_w_px * 0.8) / res, 2.0, size_x - 2.0))
        mid_x    = float(np.clip((neck_x + neck_len / 2) / res, 2.0, size_x - 2.0))
        exit_x   = float(np.clip((neck_x + neck_len - neck_w_px * 0.8) / res, 2.0, size_x - 2.0))
        goal_x   = float(np.clip((rr_x + room_sz * 0.7) / res, 2.0, size_x - 2.0))
        cy_m     = float(center_y / res)

        waypoints = np.array([
            [start_x, cy_m],
            [entry_x, cy_m],
            [mid_x,   cy_m],
            [exit_x,  cy_m],
            [goal_x,  cy_m],
        ], dtype=np.float32)

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 3: Fork Trap (Anti–Local-Minima)
# ---------------------------------------------------------------------------

class ForkTrapMap:
    """
    T-junction where one branch ends in a dead-end, the other leads to the goal.
    The bot spawns at the stem. Goal is deep in the correct branch.

    This tests whether the robot can make the correct choice at the fork rather
    than entering the dead-end branch (which initially has a similar heading error
    to the correct branch due to the symmetry of the junction).

    Note: an MLP without memory cannot reliably recover once it enters the wrong
    branch (no reverse gear). This map is therefore weighted LOW (10%) so it
    provides training signal without dominating and tanking success rate.

    Generation:
      - Stem corridor leading to junction.
      - Left branch: dead end after random_depth metres.
      - Right branch: continues to goal room.
      - Waypoints guide the robot to the junction, then directly to the goal
        (skipping the dead-end branch entirely). The agent must learn to follow
        the gradient rather than explore symmetrically.
    """

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res)
        h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corr_px   = int(2.0 * res)
        half      = corr_px // 2

        # --- Stem ---
        stem_len  = int(random.uniform(8.0, 14.0) * res)
        stem_x0   = int(0.1 * w)
        stem_y    = h // 2
        stem_x1   = stem_x0 + stem_len
        _carve(grid, stem_y - half, stem_y + half, stem_x0, stem_x1)

        # --- Junction room (small square to make the fork clear) ---
        junc_sz   = corr_px * 2
        junc_x    = stem_x1
        junc_y    = stem_y - junc_sz // 2
        _carve(grid, junc_y, junc_y + junc_sz, junc_x, junc_x + junc_sz)

        # Randomly choose which branch (top or bottom) leads to goal
        goal_is_top = random.random() > 0.5

        branch_len   = int(random.uniform(8.0, 13.0) * res)
        dead_end_len = int(random.uniform(5.0, 9.0) * res)

        # Top branch
        top_start_x = junc_x
        top_y = stem_y - junc_sz // 2 - half
        _carve(grid, top_y - half, top_y + half,
               top_start_x, top_start_x + (branch_len if goal_is_top else dead_end_len))

        # Bottom branch
        bot_y = stem_y + junc_sz // 2 + half
        _carve(grid, bot_y - half, bot_y + half,
               junc_x, junc_x + (dead_end_len if goal_is_top else branch_len))

        # Goal room at end of correct branch
        goal_room_sz = int(4.0 * res)
        if goal_is_top:
            goal_x = top_start_x + branch_len
            goal_y = top_y - goal_room_sz // 2
        else:
            goal_x = junc_x + branch_len
            goal_y = bot_y - goal_room_sz // 2

        _carve(grid,
               max(0, goal_y), min(h, goal_y + goal_room_sz),
               goal_x, min(w - 5, goal_x + goal_room_sz))

        _seal_border(grid)

        # Waypoints: stem start → junction → goal room centre
        start_wx = float(np.clip((stem_x0 + int(1.0 * res)) / res, 2.0, size_x - 2.0))
        junc_wx  = float(np.clip((junc_x + junc_sz // 2) / res, 2.0, size_x - 2.0))
        goal_wx  = float(np.clip((goal_x + goal_room_sz // 2) / res, 2.0, size_x - 2.0))
        goal_wy  = float(np.clip((goal_y + goal_room_sz // 2) / res, 2.0, size_y - 2.0))
        junc_wy  = float(np.clip(stem_y / res, 2.0, size_y - 2.0))

        waypoints = np.array([
            [start_wx,  junc_wy],
            [junc_wx,   junc_wy],
            [goal_wx,   goal_wy],
        ], dtype=np.float32)

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 4: Clutter Corridor
# ---------------------------------------------------------------------------

class ClutterCorridorMap:
    """
    A wide corridor (4–6m) filled with randomly placed pillars, forcing
    the robot to weave through obstacles at close range.

    This tests pure reactive avoidance at moderate speed — the scenario where
    the velocity budget must trim speed continuously rather than only at named
    corners. Pillar density scales with a random difficulty factor so the map
    is not trivially easy or always impassable.

    Pillars are rejection-sampled so they never block a 1.5m-wide centre line
    between start and goal (guaranteeing at least one valid path).
    """

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res)
        h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        # Wide corridor running left-to-right
        corridor_h_m = random.uniform(6.0, 9.0)
        corridor_px  = int(corridor_h_m * res)
        corr_y0 = (h - corridor_px) // 2
        _carve(grid, corr_y0, corr_y0 + corridor_px, int(0.05 * w), int(0.95 * w))

        _seal_border(grid)

        # Pillar properties
        n_pillars   = random.randint(10, 18)
        p_min_m     = 1.2
        p_max_m     = 2.0
        centre_band = int(0.75 * res)  # ±0.75m band around corridor centreline — keep clear
        centre_y    = h // 2

        start_x_m = 3.0
        goal_x_m  = size_x - 3.0

        placed = 0
        attempts = 0
        while placed < n_pillars and attempts < 300:
            attempts += 1
            p_w = int(random.uniform(p_min_m, p_max_m) * res)
            p_h = int(random.uniform(p_min_m, p_max_m) * res)

            px = random.randint(int(0.08 * w), int(0.92 * w) - p_w)
            py = random.randint(corr_y0 + int(0.3 * res),
                                corr_y0 + corridor_px - p_h - int(0.3 * res))

            cx_m = (px + p_w / 2) / res
            cy_m = (py + p_h / 2) / res

            # Reject if pillar overlaps the keep-clear centre band
            if abs((py + p_h / 2) - centre_y) < centre_band + p_h // 2:
                continue
            # Reject if too close to start or goal
            if abs(cx_m - start_x_m) < 2.5 or abs(cx_m - goal_x_m) < 2.5:
                continue

            grid[py:py + p_h, px:px + p_w] = 1
            placed += 1

        # Simple 3-waypoint path along corridor centreline
        mid_x = size_x / 2 + random.uniform(-2.0, 2.0)
        mid_y = centre_y / res + random.uniform(-0.5, 0.5)

        waypoints = np.array([
            [start_x_m, centre_y / res],
            [mid_x,     mid_y],
            [goal_x_m,  centre_y / res],
        ], dtype=np.float32)

        return grid, waypoints, res



# ---------------------------------------------------------------------------
# Map 5: Blind Corner (V5 - Inside/Outside Track Only)
# ---------------------------------------------------------------------------
class BlindCornerMap:
    """
    Z-shaped corridor with pillars flush against walls (no center placement).
    Forces the agent to check the specific side of the hallway after a turn.
    """
    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w, h = int(size_x * res), int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        # 3.0m corridor + 1.0m pillar = 2.0m clear path (exceeds 1.5m requirement)
        corr_px = int(3.0 * res)
        half = corr_px // 2

        y0, y1 = int(0.25 * h), int(0.75 * h)
        x_turn = int(0.7 * w)
        
        _carve(grid, y0 - half, y0 + half, int(0.1 * w), x_turn + half)
        _carve(grid, y0 - half, y1 + half, x_turn - half, x_turn + half)
        _carve(grid, y1 - half, y1 + half, int(0.3 * w), x_turn + half)

        block_sz = int(1.0 * res)

        # --- AMBUSH 1 (First Turn) ---
        gap1_px = int(random.uniform(0.5, 1.5) * res)
        amb1_y_s = y0 + half + gap1_px
        amb1_y_e = amb1_y_s + block_sz
        
        # Binary choice: Inside vs Outside
        if random.random() > 0.5: # Inside (Right wall of vertical chute)
            grid[amb1_y_s:amb1_y_e, x_turn - half : x_turn - half + block_sz] = 1
        else:                     # Outside (Left wall of vertical chute)
            grid[amb1_y_s:amb1_y_e, x_turn + half - block_sz : x_turn + half] = 1

        # --- AMBUSH 2 (Second Turn) ---
        gap2_px = int(random.uniform(0.5, 1.5) * res)
        amb2_x_e = x_turn - half - gap2_px
        amb2_x_s = amb2_x_e - block_sz

        if random.random() > 0.5: # Inside (Top wall)
            grid[y1 - half : y1 - half + block_sz, amb2_x_s:amb2_x_e] = 1
        else:                     # Outside (Bottom wall)
            grid[y1 + half - block_sz : y1 + half, amb2_x_s:amb2_x_e] = 1

        _seal_border(grid)

        waypoints = np.array([
            [0.15 * size_x, 0.25 * size_y],
            [0.70 * size_x, 0.25 * size_y],
            [0.70 * size_x, 0.75 * size_y],
            [0.35 * size_x, 0.75 * size_y],
        ], dtype=np.float32)

        # --- Randomization Flips ---
        if random.choice([True, False]): # Horizontal
            grid = np.fliplr(grid)
            waypoints[:, 0] = size_x - waypoints[:, 0]
        if random.choice([True, False]): # Vertical
            grid = np.flipud(grid)
            waypoints[:, 1] = size_y - waypoints[:, 1]

        return grid, waypoints, res



# ---------------------------------------------------------------------------
# FineTuneMapBank
# ---------------------------------------------------------------------------

class FineTuneMapBank:
    """
    Map provider for the finetuning phase.

    Sampling weights (must sum to 1.0):
      CornerGauntlet  35% — highest weight; the primary failure mode
      PinchPoint      30% — second most common failure
      ClutterCorridor 20% — general reactive avoidance
      ForkTrap        10% — hard, low weight to avoid tanking success rate
      BlindCorner     5% — dynamic ambush scenario
      Standard maps   remaining 30% of all draws — retention of base policy

    Note: the 70/30 split is achieved by first rolling whether to use a
    specialised map (p=0.70), then sampling within specialised types.
    """

    _GENERATORS = [
        CornerGauntletMap.generate,
        PinchPointMap.generate,
        ClutterCorridorMap.generate,
        ForkTrapMap.generate,
        BlindCornerMap.generate,
    ]
    # Weights for the specialised sub-sample (must sum to 1.0)
    _WEIGHTS = [0.25, 0.25, 0.20, 0.10, 0.20]

    def __init__(self, standard_map_dir: str = "training_maps",
                 standard_difficulty: float = 0.75):
        from MapBank import MapBank
        self._standard = MapBank(dataset_dir=standard_map_dir)
        self._standard.set_difficulty(standard_difficulty)

    def set_difficulty(self, difficulty: float):
        """
        Allows the curriculum callback to call set_difficulty without breaking.
        Difficulty is forwarded to the standard bank only; specialised maps
        have fixed geometry.
        """
        self._standard.set_difficulty(float(difficulty))

    def get_random_map(self):
        """
        Returns (map_grid: np.ndarray, waypoints: np.ndarray, resolution: int).
        Compatible with PeopleBotEnv's expected interface.
        """
        if random.random() < 0.70:
            gen = random.choices(self._GENERATORS, weights=self._WEIGHTS, k=1)[0]
            return gen()
        else:
            return self._standard.get_random_map()



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Generating visual preview of FineTune maps...")

    # The six specialized generators
    maps_to_test = [
        ("Corner Gauntlet", CornerGauntletMap),
        ("Pinch Point", PinchPointMap),
        ("Clutter Corridor", ClutterCorridorMap),
        ("Fork Trap", ForkTrapMap),
        ("Blind Corner", BlindCornerMap),
    ]

    # Expanded to 2x3 to fit all maps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (name, map_class) in zip(axes, maps_to_test):
        # 1. Generate the map
        grid, waypoints, res = map_class.generate(size_x=40, size_y=40)

        # 2. Convert grid array dimensions into physical meters for the plot extent
        h_px, w_px = grid.shape
        w_m = w_px / res
        h_m = h_px / res

        # 3. Plot the grid (Inverted so 0=Free Space=White, 1=Walls=Black)
        ax.imshow(grid == 0, cmap='gray', origin='lower', extent=[0, w_m, 0, h_m])

        # 4. Overlay the waypoints and path lines
        if len(waypoints) > 0:
            wx = waypoints[:, 0]
            wy = waypoints[:, 1]

            # Draw the intended path
            ax.plot(wx, wy, color='red', linestyle='--', alpha=0.8, linewidth=2, label="Planned Path")
            
            # Draw the intermediate waypoints
            ax.scatter(wx, wy, c='yellow', s=60, zorder=5)

            # Mark Start (Lime) and Goal (Blue X)
            ax.scatter(wx[0], wy[0], c='lime', s=120, zorder=6, edgecolors='black', label="Start")
            ax.scatter(wx[-1], wy[-1], c='blue', s=120, marker='X', zorder=6, edgecolors='black', label="Goal")

        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.legend(loc="upper right", fontsize=9)
        
        # Dark background for any out-of-bounds rendering
        ax.set_facecolor('#222222')

    plt.tight_layout()
    plt.show()
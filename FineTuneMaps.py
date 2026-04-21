"""
FineTuneMaps.py
===============
Specialized map generators for the corner/narrow-corridor finetuning phase.

BUG FIX: CornerGauntlet waypoint ordering after geometric transforms
────────────────────────────────────────────────────────────────────
The transpose + flip combination could place waypoints[0] (spawn) at a
high-x, high-y corner with the first waypoint demanding a 25–30m commitment
westward before any progress reward is credited. With the heading penalty
and velocity budget both active, this caused the bot to stall at spawn:
the stop-oscillation trap (velocity penalty fires → bot stops → existence
tax → bot stays stopped) dominated the weak progress gradient over such
a long first leg.

Fix: after all transforms, if the last waypoint is closer to the map origin
than the first, reverse the waypoint array. This is geometrically neutral
(same map, same path, just traversed in opposite direction) but ensures the
spawn is always at the structurally simpler end, keeping the first leg short
enough that the progress gradient dominates from step 1.

CornerGauntlet path-length controls (unchanged from previous analysis)
────────────────────────────────────────────────────────────────────────
n_turns: randint(2,4) → randint(2,3)
Vertical excursion: 0.22h–0.38h / 0.62h–0.78h → 0.28h–0.38h / 0.62h–0.72h
"""

import numpy as np
import random

RESOLUTION = 50  # pixels per meter


def _carve(grid, y_lo, y_hi, x_lo, x_hi):
    h, w = grid.shape
    y_lo = max(0, y_lo);  y_hi = min(h, y_hi)
    x_lo = max(0, x_lo);  x_hi = min(w, x_hi)
    if y_hi > y_lo and x_hi > x_lo:
        grid[y_lo:y_hi, x_lo:x_hi] = 0


def _seal_border(grid, border_px=5):
    grid[:border_px, :] = 1;  grid[-border_px:, :] = 1
    grid[:, :border_px] = 1;  grid[:, -border_px:] = 1


# ---------------------------------------------------------------------------
# Map 1: Corner Gauntlet
# ---------------------------------------------------------------------------

class CornerGauntletMap:
    """
    Zigzag path of 2–3 right-angle turns through 2m-wide corridors.
    Tests velocity reduction at 90° corners.
    """

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res);  h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corr_px = int(2.0 * res);  half = corr_px // 2
        n_turns = random.randint(2, 3)      # was randint(2, 4)
        margin  = int(3.5 * res)

        pts = [(margin, h // 2)]
        x_step = (w - 2 * margin) // (n_turns + 1)

        for i in range(n_turns):
            x = margin + (i + 1) * x_step
            if i % 2 == 0:
                y = random.randint(int(0.28 * h), int(0.38 * h))  # was 0.22
            else:
                y = random.randint(int(0.62 * h), int(0.72 * h))  # was 0.78
            pts.append((x, y))

        pts.append((w - margin, h // 2))

        for i in range(len(pts) - 1):
            x0, y0 = pts[i];  x1, y1 = pts[i + 1]
            _carve(grid, y0 - half, y0 + half,
                   min(x0, x1) - half, max(x0, x1) + half)
            _carve(grid, min(y0, y1) - half, max(y0, y1) + half,
                   x1 - half, x1 + half)

        _seal_border(grid)

        waypoints = np.array(
            [[float(np.clip(px / res, 2.0, size_x - 2.0)),
              float(np.clip(py / res, 2.0, size_y - 2.0))]
             for px, py in pts],
            dtype=np.float32
        )

        # Geometric transforms
        if random.choice([True, False]):
            grid = np.fliplr(grid)
            waypoints[:, 0] = size_x - waypoints[:, 0]
        if random.choice([True, False]):
            grid = np.flipud(grid)
            waypoints[:, 1] = size_y - waypoints[:, 1]
        if random.choice([True, False]):
            grid = grid.T
            waypoints[:, [0, 1]] = waypoints[:, [1, 0]]

        # ── Waypoint ordering fix ─────────────────────────────────────────────
        # Ensure spawn (waypoints[0]) is at the end closer to the map origin,
        # so the first leg is the shortest possible commitment. The map geometry
        # is identical either way — only the traversal direction changes.
        start = waypoints[0];  end = waypoints[-1]
        if float(np.hypot(end[0], end[1])) < float(np.hypot(start[0], start[1])):
            waypoints = waypoints[::-1].copy()

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 2: Pinch Point
# ---------------------------------------------------------------------------

class PinchPointMap:
    """Wide room → 1.5–1.8m neck → wide room. Tests narrow-passage commitment."""

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res);  h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        center_y  = h // 2
        neck_w_px = int(random.uniform(1.5, 1.8) * res)
        neck_len  = int(random.uniform(5.0, 9.0) * res)
        room_sz   = int(random.uniform(6.0, 9.0) * res)
        neck_x    = w // 2 - neck_len // 2
        neck_y    = center_y - neck_w_px // 2

        lr_x = max(int(0.05 * w), neck_x - room_sz)
        _carve(grid, center_y - room_sz // 2, center_y + room_sz // 2,
               lr_x, neck_x + neck_w_px)
        _carve(grid, neck_y, neck_y + neck_w_px, neck_x, neck_x + neck_len)

        rr_x = neck_x + neck_len
        _carve(grid, center_y - room_sz // 2, center_y + room_sz // 2,
               rr_x, min(int(0.95 * w), rr_x + room_sz))
        _seal_border(grid)

        cy = float(center_y / res)
        waypoints = np.array([
            [float(np.clip((lr_x + room_sz * 0.3)                / res, 2.0, size_x - 2.0)), cy],
            [float(np.clip((neck_x + neck_w_px * 0.8)            / res, 2.0, size_x - 2.0)), cy],
            [float(np.clip((neck_x + neck_len / 2)               / res, 2.0, size_x - 2.0)), cy],
            [float(np.clip((neck_x + neck_len - neck_w_px * 0.8) / res, 2.0, size_x - 2.0)), cy],
            [float(np.clip((rr_x + room_sz * 0.7)                / res, 2.0, size_x - 2.0)), cy],
        ], dtype=np.float32)

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 3: Fork Trap
# ---------------------------------------------------------------------------

class ForkTrapMap:
    """T-junction: one dead-end branch, one goal branch. Weighted LOW (10%)."""

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res);  h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corr_px = int(2.0 * res);  half = corr_px // 2
        stem_x0 = int(0.1 * w);  stem_y = h // 2
        stem_x1 = stem_x0 + int(random.uniform(8.0, 14.0) * res)
        _carve(grid, stem_y - half, stem_y + half, stem_x0, stem_x1)

        junc_sz = corr_px * 2;  junc_x = stem_x1
        _carve(grid, stem_y - junc_sz // 2, stem_y + junc_sz // 2,
               junc_x, junc_x + junc_sz)

        goal_is_top  = random.random() > 0.5
        branch_len   = int(random.uniform(8.0, 13.0) * res)
        dead_end_len = int(random.uniform(5.0,  9.0) * res)

        top_y = stem_y - junc_sz // 2 - half
        _carve(grid, top_y - half, top_y + half,
               junc_x, junc_x + (branch_len if goal_is_top else dead_end_len))
        bot_y = stem_y + junc_sz // 2 + half
        _carve(grid, bot_y - half, bot_y + half,
               junc_x, junc_x + (dead_end_len if goal_is_top else branch_len))

        goal_room_sz = int(4.0 * res)
        goal_y_base  = top_y if goal_is_top else bot_y
        goal_x       = junc_x + branch_len
        goal_y       = goal_y_base - goal_room_sz // 2
        _carve(grid, max(0, goal_y), min(h, goal_y + goal_room_sz),
               goal_x, min(w - 5, goal_x + goal_room_sz))
        _seal_border(grid)

        junc_wy = float(np.clip(stem_y / res, 2.0, size_y - 2.0))
        waypoints = np.array([
            [float(np.clip((stem_x0 + int(res))           / res, 2.0, size_x - 2.0)), junc_wy],
            [float(np.clip((junc_x + junc_sz // 2)        / res, 2.0, size_x - 2.0)), junc_wy],
            [float(np.clip((goal_x + goal_room_sz // 2)   / res, 2.0, size_x - 2.0)),
             float(np.clip((goal_y + goal_room_sz // 2)   / res, 2.0, size_y - 2.0))],
        ], dtype=np.float32)

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 4: Clutter Corridor
# ---------------------------------------------------------------------------

class ClutterCorridorMap:
    """Wide corridor (6–9m) with rejection-sampled pillars. Centre line kept clear."""

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res);  h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corridor_px = int(random.uniform(6.0, 9.0) * res)
        corr_y0     = (h - corridor_px) // 2
        _carve(grid, corr_y0, corr_y0 + corridor_px, int(0.05 * w), int(0.95 * w))
        _seal_border(grid)

        centre_y = h // 2;  centre_band = int(0.75 * res)
        start_x_m = 3.0;    goal_x_m    = size_x - 3.0

        placed = 0;  attempts = 0
        while placed < random.randint(10, 18) and attempts < 300:
            attempts += 1
            p_w = int(random.uniform(1.2, 2.0) * res)
            p_h = int(random.uniform(1.2, 2.0) * res)
            px  = random.randint(int(0.08 * w), int(0.92 * w) - p_w)
            py  = random.randint(corr_y0 + int(0.3 * res),
                                 corr_y0 + corridor_px - p_h - int(0.3 * res))
            if abs((py + p_h / 2) - centre_y) < centre_band + p_h // 2: continue
            cx_m = (px + p_w / 2) / res
            if abs(cx_m - start_x_m) < 2.5 or abs(cx_m - goal_x_m) < 2.5: continue
            grid[py:py + p_h, px:px + p_w] = 1
            placed += 1

        cy_m = centre_y / res
        mid_x = size_x / 2 + random.uniform(-2.0, 2.0)
        waypoints = np.array([
            [start_x_m, cy_m],
            [mid_x,     cy_m + random.uniform(-0.5, 0.5)],
            [goal_x_m,  cy_m],
        ], dtype=np.float32)

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# Map 5: Blind Corner
# ---------------------------------------------------------------------------

class BlindCornerMap:
    """Z-shaped corridor with ambush pillars flush against walls after each turn."""

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w, h = int(size_x * res), int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corr_px = int(3.0 * res);  half = corr_px // 2
        y0 = int(0.25 * h);  y1 = int(0.75 * h);  x_turn = int(0.7 * w)

        _carve(grid, y0 - half, y0 + half, int(0.1 * w), x_turn + half)
        _carve(grid, y0 - half, y1 + half, x_turn - half, x_turn + half)
        _carve(grid, y1 - half, y1 + half, int(0.3 * w), x_turn + half)

        block_sz = int(1.0 * res)
        gap1 = int(random.uniform(0.5, 1.5) * res)
        s1   = y0 + half + gap1;  e1 = s1 + block_sz
        if random.random() > 0.5:
            grid[s1:e1, x_turn - half : x_turn - half + block_sz] = 1
        else:
            grid[s1:e1, x_turn + half - block_sz : x_turn + half] = 1

        gap2 = int(random.uniform(0.5, 1.5) * res)
        xe   = x_turn - half - gap2;  xs = xe - block_sz
        if random.random() > 0.5:
            grid[y1 - half : y1 - half + block_sz, xs:xe] = 1
        else:
            grid[y1 + half - block_sz : y1 + half, xs:xe] = 1

        _seal_border(grid)

        waypoints = np.array([
            [0.15 * size_x, 0.25 * size_y],
            [0.70 * size_x, 0.25 * size_y],
            [0.70 * size_x, 0.75 * size_y],
            [0.35 * size_x, 0.75 * size_y],
        ], dtype=np.float32)

        if random.choice([True, False]):
            grid = np.fliplr(grid);  waypoints[:, 0] = size_x - waypoints[:, 0]
        if random.choice([True, False]):
            grid = np.flipud(grid);  waypoints[:, 1] = size_y - waypoints[:, 1]

        return grid, waypoints, res


# ---------------------------------------------------------------------------
# FineTuneMapBank
# ---------------------------------------------------------------------------

class FineTuneMapBank:
    """
    70% specialised / 30% standard maps.
    Specialised weights: Gauntlet 35%, Pinch 30%, Clutter 20%, Fork 10%, Blind 5%.
    """

    _GENERATORS = [
        CornerGauntletMap.generate,
        PinchPointMap.generate,
        ClutterCorridorMap.generate,
        ForkTrapMap.generate,
        BlindCornerMap.generate,
    ]
    _WEIGHTS = [0.35, 0.30, 0.20, 0.10, 0.05]

    def __init__(self, standard_map_dir="training_maps", standard_difficulty=0.75):
        from MapBank import MapBank
        self._standard = MapBank(dataset_dir=standard_map_dir)
        self._standard.set_difficulty(standard_difficulty)

    def set_difficulty(self, difficulty):
        self._standard.set_difficulty(float(difficulty))

    def get_random_map(self):
        if random.random() < 0.70:
            return random.choices(self._GENERATORS, weights=self._WEIGHTS, k=1)[0]()
        return self._standard.get_random_map()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    maps_to_test = [
        ("Corner Gauntlet",  CornerGauntletMap),
        ("Pinch Point",      PinchPointMap),
        ("Clutter Corridor", ClutterCorridorMap),
        ("Fork Trap",        ForkTrapMap),
        ("Blind Corner",     BlindCornerMap),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (name, cls) in zip(axes.flatten(), maps_to_test):
        grid, wps, res = cls.generate()
        h_px, w_px = grid.shape
        ax.imshow(grid == 0, cmap='gray', origin='lower',
                  extent=[0, w_px / res, 0, h_px / res])
        ax.plot(wps[:, 0], wps[:, 1], 'r--', lw=2, alpha=0.8)
        ax.scatter(wps[:, 0], wps[:, 1], c='yellow', s=60, zorder=5)
        ax.scatter(wps[0, 0],  wps[0, 1],  c='lime', s=120, zorder=6, edgecolors='k')
        ax.scatter(wps[-1, 0], wps[-1, 1], c='blue', s=120, marker='X', zorder=6, edgecolors='k')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_facecolor('#222222')
    plt.tight_layout();  plt.show()
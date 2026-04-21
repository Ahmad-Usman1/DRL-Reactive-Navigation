"""
FineTuneMaps.py  (Phase 2 — Multi-Difficulty Curriculum)
=========================================================
Extends FineTuneMapBank to sample from three difficulty bins within the
existing training_maps/ directory.

MAPBANK BIN DISCOVERY (corrected — matches actual MapBank.py)
─────────────────────────────────────────────────────────────
MapBank stores maps in per-difficulty subfolders:
    training_maps/diff_0.00/  diff_0.10/  diff_0.25/  diff_0.50/
                  diff_0.75/  diff_0.80/  diff_1.00/
set_difficulty(x) snaps to the NEAREST existing bin.

Therefore the three difficulty banks below all point at the SAME dataset_dir.
They differ only in their stored set_difficulty target. Memory cost is minimal
(MapBank is lazy-loading — only file paths live in RAM, ~1MB per instance per
the MapBank docstring).

DIFFICULTY POOL WEIGHTS PER CURRICULUM STAGE:
    Stage 1  (first 30% of training): 0.60 / 0.40 / 0.00  (ease into 0.8)
    Stage 2  (next 40%):               0.30 / 0.50 / 0.20  (introduce 1.0)
    Stage 3  (final 30%):              0.20 / 0.40 / 0.40  (full mix)

The Finetune_V3 training script advances stages via CurriculumStageCallback.

SPECIALIZED MAP GENERATORS (unchanged)
───────────────────────────────────────
CornerGauntlet/PinchPoint/ClutterCorridor/ForkTrap/BlindCorner are identical
to the previous version. The 70/30 specialized-vs-standard split is preserved.
"""

import numpy as np
import random

RESOLUTION = 50


def _carve(grid, y_lo, y_hi, x_lo, x_hi):
    h, w = grid.shape
    y_lo = max(0, y_lo);  y_hi = min(h, y_hi)
    x_lo = max(0, x_lo);  x_hi = min(w, x_hi)
    if y_hi > y_lo and x_hi > x_lo:
        grid[y_lo:y_hi, x_lo:x_hi] = 0


def _seal_border(grid, border_px=5):
    grid[:border_px, :] = 1;  grid[-border_px:, :] = 1
    grid[:, :border_px] = 1;  grid[:, -border_px:] = 1


# ─── Corner Gauntlet ────────────────────────────────────────────────────────

class CornerGauntletMap:
    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res);  h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        corr_px = int(2.0 * res);  half = corr_px // 2
        n_turns = random.randint(2, 3)
        margin  = int(3.5 * res)

        pts = [(margin, h // 2)]
        x_step = (w - 2 * margin) // (n_turns + 1)
        for i in range(n_turns):
            x = margin + (i + 1) * x_step
            if i % 2 == 0:
                y = random.randint(int(0.28 * h), int(0.38 * h))
            else:
                y = random.randint(int(0.62 * h), int(0.72 * h))
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

        if random.choice([True, False]):
            grid = np.fliplr(grid);  waypoints[:, 0] = size_x - waypoints[:, 0]
        if random.choice([True, False]):
            grid = np.flipud(grid);  waypoints[:, 1] = size_y - waypoints[:, 1]
        if random.choice([True, False]):
            grid = grid.T;  waypoints[:, [0, 1]] = waypoints[:, [1, 0]]

        start = waypoints[0];  end = waypoints[-1]
        if float(np.hypot(end[0], end[1])) < float(np.hypot(start[0], start[1])):
            waypoints = waypoints[::-1].copy()

        return grid, waypoints, res


# ─── Pinch Point ────────────────────────────────────────────────────────────

class PinchPointMap:
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


# ─── Fork Trap ──────────────────────────────────────────────────────────────

class ForkTrapMap:
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


# ─── Clutter Corridor ───────────────────────────────────────────────────────

class ClutterCorridorMap:
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


# ─── Offset Clutter (Goal-Heading Decision Teacher) ──────────────────────────

class OffsetClutterMap:
    """
    Wide corridor with centreline obstruction forcing left/right detour choice.
    The goal is positioned such that the narrow path aligns with goal heading,
    but the wide detour requires heading away from the goal. Waypoints guide
    the bot along the safe (wide) path.

    Teaches: clearance signal must outweigh heading penalty when the heading-
    aligned path is impassable or dangerously tight.

    DESIGN PRINCIPLE
    ────────────────
    The bot receives heading feedback toward the current waypoint (not final goal).
    If waypoints follow the wide path but the goal sits past the narrow gap,
    then:
      - Following waypoints → heading away from goal initially → heading penalty fires
      - Ignoring waypoints, cutting to goal → heading aligned → but tight crash risk

    With correct reward weighting (clearance outweighing heading), the bot learns
    waypoint > goal heading. With broken reward (heading > clearance), it crashes
    trying to cut through the narrow path.

    This map is diagnostic: bot performance here reveals whether the reward
    priorities are correct.
    """

    @staticmethod
    def generate(size_x=40, size_y=40):
        res = RESOLUTION
        w = int(size_x * res);  h = int(size_y * res)
        grid = np.ones((h, w), dtype=np.int8)

        # Wide main corridor
        corr_width_px = int(3.5 * res)  # 3.5m wide — comfortable
        corr_y0 = (h - corr_width_px) // 2
        _carve(grid, corr_y0, corr_y0 + corr_width_px, int(0.05 * w), int(0.95 * w))

        # Centreline obstruction (1.5m wide, extends from top or bottom into corridor)
        obs_width_px = int(1.5 * res)
        obs_depth_px = int(random.uniform(2.0, 3.5) * res)

        # Obstruction placed at x=0.55 (halfway through corridor, slightly right of center)
        obs_x = int(0.55 * w)

        # Choose top or bottom placement randomly
        if random.random() > 0.5:
            # Top placement: obstruction blocks top half, forces detour to bottom
            obs_y0 = corr_y0 - obs_depth_px
            obs_y1 = corr_y0 + int(corr_width_px * 0.4)
            tight_side = "bottom"
            wide_side = "top"
        else:
            # Bottom placement: obstruction blocks bottom half, forces detour to top
            obs_y0 = corr_y0 + int(corr_width_px * 0.6)
            obs_y1 = corr_y0 + corr_width_px + obs_depth_px
            tight_side = "top"
            wide_side = "bottom"

        _carve(grid, max(0, obs_y0), min(h, obs_y1), obs_x, obs_x + obs_width_px)

        # Goal placement: past the obstruction, on the tight-side to make it tempting
        goal_x = int(0.85 * w)
        if tight_side == "bottom":
            goal_y = corr_y0 + int(corr_width_px * 0.7)
        else:
            goal_y = corr_y0 + int(corr_width_px * 0.3)

        # Seal borders
        _seal_border(grid)

        # Waypoints: guide bot along the WIDE detour
        # Waypoint 1: start
        wp1_x = float(0.1 * size_x)
        wp1_y = float(size_y / 2.0)

        # Waypoint 2: pre-obstruction (neutral, before decision point)
        wp2_x = float(0.35 * size_x)
        wp2_y = float(size_y / 2.0)

        # Waypoint 3: through the wide detour (forces heading away from goal initially)
        if wide_side == "top":
            wp3_y = float(0.25 * size_y)
        else:
            wp3_y = float(0.75 * size_y)
        wp3_x = float(0.60 * size_x)

        # Waypoint 4: goal (past obstruction on tight side, but now approached safely)
        wp4_x = float(0.85 * size_x)
        wp4_y = float(goal_y / res)

        waypoints = np.array([
            [wp1_x, wp1_y],
            [wp2_x, wp2_y],
            [wp3_x, wp3_y],
            [wp4_x, wp4_y],
        ], dtype=np.float32)

        # Random transforms (keep consistency with other maps)
        if random.choice([True, False]):
            grid = np.fliplr(grid);  waypoints[:, 0] = size_x - waypoints[:, 0]
        if random.choice([True, False]):
            grid = np.flipud(grid);  waypoints[:, 1] = size_y - waypoints[:, 1]

        return grid, waypoints, res


# ─── Blind Corner ───────────────────────────────────────────────────────────

class BlindCornerMap:
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


# ─── FineTuneMapBank with Curriculum ────────────────────────────────────────

class FineTuneMapBank:
    """
    Curriculum-aware map bank.

    70% specialized (Gauntlet/Pinch/Clutter/Fork/Blind).
    30% standard, drawn from three difficulty bins in training_maps/ according
    to the current curriculum stage.

    All three difficulty banks share the same dataset_dir. They are separate
    MapBank instances only so each can hold its own set_difficulty() target
    without mutual interference during concurrent access.
    """

    _SPECIALIZED_GENERATORS = [
        CornerGauntletMap.generate,
        PinchPointMap.generate,
        ClutterCorridorMap.generate,
        OffsetClutterMap.generate,
        ForkTrapMap.generate,
        BlindCornerMap.generate,
    ]
    _SPECIALIZED_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]

    # Difficulty pool weights per curriculum stage: [0.75, 0.80, 1.00]
    #
    # PHASE 2a — NO diff=1.00 EXPOSURE.
    # The policy crashed at 30% when diff=1.00 entered at only 6% of total maps
    # (30% standard draw × 20% weight). 1.5m corridors are 30% tighter than
    # anything previously trained on — a distribution gap the policy cannot bridge
    # in a single step. Phase 2a consolidates at diff=0.80 only.
    #
    # Stage 1 (first 50% of training): ease from 0.75 toward 0.80
    # Stage 2 (final 50%):             fully consolidate at 0.80
    #
    # Phase 2b (separate run, after 2a success criteria are met) will introduce
    # diff=1.00 starting from the stable 2a checkpoint.
    _STAGE_WEIGHTS = {
        1: [0.70, 0.30, 0.00],
        2: [0.40, 0.60, 0.00],
        3: [0.40, 0.60, 0.00],   # unused in Phase 2a; mirrors Stage 2 as a safe fallback
    }

    def __init__(self,
                 standard_map_dir: str = "training_maps",
                 standard_difficulty: float = 0.75):
        """
        standard_difficulty is a legacy parameter. It now routes to the
        curriculum stage via set_difficulty(). Default 0.75 → stage 1.
        """
        from MapBank import MapBank

        # Three banks sharing the same dataset_dir. MapBank will snap each
        # set_difficulty target to the nearest available bin:
        #   0.75 → diff_0.75  (exists)
        #   0.80 → diff_0.80  (created by GenerateDifficulty08Maps.py)
        #   1.00 → diff_1.00  (exists)
        self._bank_075 = MapBank(dataset_dir=standard_map_dir)
        self._bank_075.set_difficulty(0.75)

        self._bank_080 = MapBank(dataset_dir=standard_map_dir)
        self._bank_080.set_difficulty(0.80)

        self._bank_100 = MapBank(dataset_dir=standard_map_dir)
        self._bank_100.set_difficulty(1.00)

        # Detect whether diff=0.80 bin actually exists. MapBank's closest-bin
        # snap is silent — without this check we'd unknowingly fall back to
        # 0.75 or 1.00 when diff=0.80 was requested.
        self._has_080 = 0.80 in self._bank_080.binned_paths
        if not self._has_080:
            print("[FineTuneMapBank] WARNING: diff_0.80 bin not found in "
                  f"{standard_map_dir}. Run GenerateDifficulty08Maps.py first.")
            print("[FineTuneMapBank] Will redistribute diff=0.80 weight to "
                  "diff=0.75 and diff=1.00 pools.")

        self._stage = 1
        self.set_difficulty(standard_difficulty)

    def set_curriculum_stage(self, stage: int):
        """Explicit stage control for the CurriculumStageCallback."""
        assert stage in (1, 2, 3), f"stage must be 1, 2, or 3; got {stage}"
        self._stage = stage

    def set_difficulty(self, difficulty: float):
        """
        Legacy compatibility. Maps scalar difficulty to curriculum stage.
        Phase 2a only uses stages 1 and 2 (no diff=1.00 exposure).
        """
        if difficulty < 0.82: self._stage = 1
        else:                  self._stage = 2

    def get_random_map(self):
        if random.random() < 0.70:
            gen = random.choices(self._SPECIALIZED_GENERATORS,
                                 weights=self._SPECIALIZED_WEIGHTS, k=1)[0]
            return gen()

        weights = list(self._STAGE_WEIGHTS[self._stage])

        # If diff=0.80 bin missing, split its weight 50/50 between the other two
        if not self._has_080:
            weights[0] += weights[1] * 0.5
            weights[2] += weights[1] * 0.5
            weights[1]  = 0.0

        pool_idx = random.choices([0, 1, 2], weights=weights, k=1)[0]

        if   pool_idx == 0: return self._bank_075.get_random_map()
        elif pool_idx == 1: return self._bank_080.get_random_map()
        else:               return self._bank_100.get_random_map()
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import binary_dilation

class MapGenerator:
    
    @staticmethod
    def _is_los_clear_ds(x0, y0, x1, y1, pf_matrix):
        """Checks if a straight line between two points hits a wall in the downsampled grid."""
        dist = max(abs(x1 - x0), abs(y1 - y0))
        if dist == 0: return True
        xs = np.linspace(x0, x1, num=dist, endpoint=True).astype(int)
        ys = np.linspace(y0, y1, num=dist, endpoint=True).astype(int)
        return np.all(pf_matrix[ys, xs] == 1)

    @staticmethod
    def generate(size_x=40, size_y=40, difficulty=0.75):
        resolution = 50  
        grid_w = int(size_x * resolution)
        grid_h = int(size_y * resolution)

        # =========================================================
        # TIER 0.0 & 0.10: OPEN ROOMS (Sensor Calibration & Avoidance)
        # =========================================================
        if difficulty <= 0.15:
            grid_map = np.ones((grid_h, grid_w), dtype=np.int8)
            grid_map[5:-5, 5:-5] = 0 # Clear the center
            
            # 1. Randomize Start and End Points to prevent directional overfitting
            while True:
                sx = random.uniform(5.0, size_x - 5.0)
                sy = random.uniform(5.0, size_y - 5.0)
                gx = random.uniform(5.0, size_x - 5.0)
                gy = random.uniform(5.0, size_y - 5.0)
                # Ensure they are at least 25 meters apart to force a long drive
                if math.hypot(gx - sx, gy - sy) > 25.0:
                    break
                    
            wp_start = [sx, sy]
            wp_end = [gx, gy]
            
            # 2. Scale pillar counts. 
            # Diff 0.0 gets 2-4. Diff 0.10 gets a brutal 8-12.
            num_pillars = random.randint(2, 4) if difficulty < 0.05 else random.randint(8, 12)
            
            # 3. Generate MASSIVE variable-sized pillars
            for _ in range(num_pillars):
                w_m = random.uniform(1.5, 2.5) # Wide enough that Lidar CANNOT miss them
                h_m = random.uniform(1.5, 2.5)
                w_px = int(w_m * resolution)
                h_px = int(h_m * resolution)
                
                px = random.randint(int(grid_w*0.1), int(grid_w*0.9) - w_px)
                py = random.randint(int(grid_h*0.1), int(grid_h*0.9) - h_px)
                
                # Check to ensure we don't drop a pillar directly on the start/end points
                center_x = (px + w_px/2) / resolution
                center_y = (py + h_px/2) / resolution
                if math.hypot(center_x - sx, center_y - sy) < 3.0: continue
                if math.hypot(center_x - gx, center_y - gy) < 3.0: continue
                
                grid_map[py:py+h_px, px:px+w_px] = 1
                
            # No A* needed here. The local PPO agent must figure out how to dodge the pillars.
            return grid_map, np.array([wp_start, wp_end]), resolution

        # =========================================================
        # DYNAMIC GEOMETRIC SCALING (Diff 0.25 to 1.0)
        # =========================================================
        corridor_width_m = max(1.5, 4.16 - (2.66 * difficulty))
        corridor_px = int(round(corridor_width_m * resolution))
        
        room_min_m = max(2.5, 4.0 - (1.0 * difficulty))
        room_max_m = max(4.0, 7.0 - (2.0 * difficulty))
        room_min_px = int(round(room_min_m * resolution))
        room_max_px = int(round(room_max_m * resolution))
        
        # Clutter shrinks slightly at max diff so it doesn't plug the 1.5m hallways
        clutter_size_m = max(0.50, 0.7 - (0.2 * difficulty)) 
        clutter_px = int(round(clutter_size_m * resolution))
        
        # PUSH WAYPOINTS AWAY FROM WALLS
        safe_radius_m = 0.45 
        safe_radius_px = int(round(safe_radius_m * resolution))

        room_spawn_prob = 0.6 
        clutter_spawn_prob = min(0.40, max(0.0, (difficulty - 0.3) * 0.8))

        valid_map = False
        attempts = 0

        while not valid_map:
            attempts += 1
            if attempts > 60:
                print(f"Warning: Map Gen failed 60 times at Diff {difficulty}. Using Fallback Slalom.")
                return MapGenerator.create_fallback_slalom(size_x, size_y, resolution)

            grid_map = np.ones((grid_h, grid_w), dtype=np.int8)
            rooms = []
            sectors_x, sectors_y = 3, 3
            sector_w, sector_h = grid_w // sectors_x, grid_h // sectors_y

            # 1. Spawn Rooms
            for sx in range(sectors_x):
                for sy in range(sectors_y):
                    if random.random() < room_spawn_prob:
                        w = random.randint(room_min_px, room_max_px)
                        h = random.randint(room_min_px, room_max_px)
                        min_x = sx * sector_w + 2
                        max_x = (sx + 1) * sector_w - w - 2
                        min_y = sy * sector_h + 2
                        max_y = (sy + 1) * sector_h - h - 2
                        
                        if max_x <= min_x or max_y <= min_y: continue
                        x = random.randint(min_x, max_x)
                        y = random.randint(min_y, max_y)
                        grid_map[y:y+h, x:x+w] = 0
                        rooms.append([int(x + w/2), int(y + h/2)])

            if len(rooms) < 3: continue
            rooms.sort(key=lambda r: r[0])
            clutter_blocks = []

            # 2. Draw Corridors & Schedule Clutter
            for i in range(len(rooms) - 1):
                c1, c2 = rooms[i], rooms[i+1]
                
                # Horizontal Connection
                y_start, y_end = c1[1], c1[1] + corridor_px
                x_start, x_end = min(c1[0], c2[0]), max(c1[0], c2[0])
                y_end, x_end = min(y_end, grid_h - 1), min(x_end, grid_w - 1)
                grid_map[y_start:y_end, x_start:x_end] = 0

                if random.random() < clutter_spawn_prob:
                    ox = random.randint(x_start, x_end - clutter_px) if x_end - clutter_px > x_start else x_start
                    oy = y_start if random.random() > 0.5 else y_end - clutter_px
                    if ox + clutter_px < grid_w and oy + clutter_px < grid_h:
                        clutter_blocks.append((oy, ox)) 

                # Vertical Connection
                y_start_v, y_end_v = min(c1[1], c2[1]), max(c1[1], c2[1])
                x_start_v, x_end_v = c2[0], c2[0] + corridor_px
                y_end_v, x_end_v = min(y_end_v, grid_h - 1), min(x_end_v, grid_w - 1)
                grid_map[y_start_v:y_end_v, x_start_v:x_end_v] = 0

                if random.random() < clutter_spawn_prob:
                    oy = random.randint(y_start_v, y_end_v - clutter_px) if y_end_v - clutter_px > y_start_v else y_start_v
                    ox = x_start_v if random.random() > 0.5 else x_end_v - clutter_px
                    if ox + clutter_px < grid_w and oy + clutter_px < grid_h:
                        clutter_blocks.append((oy, ox)) 

            grid_map[0:2, :] = 1; grid_map[-2:, :] = 1
            grid_map[:, 0:2] = 1; grid_map[:, -2:] = 1

            # =========================================================
            # PASS 1: CLEAN PATHFINDING
            # =========================================================
            DS = 5 
            small_grid_clean = grid_map[::DS, ::DS]
            small_safe_radius = int(math.ceil(safe_radius_px / DS))
            inflated_clean = binary_dilation(small_grid_clean == 1, iterations=small_safe_radius)
            pf_matrix_clean = (~inflated_clean).astype(int) 
            pf_grid_clean = Grid(matrix=pf_matrix_clean.tolist())

            max_dist = 0
            best_pair = (0, len(rooms)-1)
            for r1 in range(len(rooms)):
                for r2 in range(r1 + 1, len(rooms)):
                    d = math.hypot(rooms[r1][0] - rooms[r2][0], rooms[r1][1] - rooms[r2][1])
                    if d > max_dist:
                        max_dist = d
                        best_pair = (r1, r2)
            
            start_idx, end_idx = best_pair
            s_node_clean = pf_grid_clean.node(rooms[start_idx][0] // DS, rooms[start_idx][1] // DS)
            e_node_clean = pf_grid_clean.node(rooms[end_idx][0] // DS, rooms[end_idx][1] // DS)
            
            if not s_node_clean.walkable or not e_node_clean.walkable: continue

            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path_nodes_clean, _ = finder.find_path(s_node_clean, e_node_clean, pf_grid_clean)
            if not path_nodes_clean: continue

            smoothed_nodes = [path_nodes_clean[0]]
            curr_idx = 0
            while curr_idx < len(path_nodes_clean) - 1:
                furthest_visible = curr_idx + 1
                for j in range(len(path_nodes_clean) - 1, curr_idx, -1):
                    if MapGenerator._is_los_clear_ds(path_nodes_clean[curr_idx].x, path_nodes_clean[curr_idx].y, 
                                                     path_nodes_clean[j].x, path_nodes_clean[j].y, pf_matrix_clean):
                        furthest_visible = j
                        break
                smoothed_nodes.append(path_nodes_clean[furthest_visible])
                curr_idx = furthest_visible
                
            raw_path = []
            for node in smoothed_nodes:
                wx = ((node.x * DS) + (DS / 2.0)) / resolution
                wy = ((node.y * DS) + (DS / 2.0)) / resolution
                raw_path.append([wx, wy])
            sparse_waypoints = np.array(raw_path)

            final_path = [sparse_waypoints[0]]
            for i in range(1, len(sparse_waypoints) - 1):
                p_curr = sparse_waypoints[i]
                dist_to_last = math.hypot(p_curr[0] - final_path[-1][0], p_curr[1] - final_path[-1][1])
                if dist_to_last >= 1.5: final_path.append(p_curr)
                    
            final_pt = sparse_waypoints[-1]
            dist_to_goal = math.hypot(final_pt[0] - final_path[-1][0], final_pt[1] - final_path[-1][1])
            if dist_to_goal < 1.0 and len(final_path) > 1: final_path[-1] = final_pt
            elif dist_to_goal >= 0.1: final_path.append(final_pt)
            sparse_waypoints = np.array(final_path)

            # =========================================================
            # PASS 2: CLUTTER VALIDATION
            # =========================================================
            for oy, ox in clutter_blocks:
                grid_map[oy:oy+clutter_px, ox:ox+clutter_px] = 1

            small_grid_cluttered = grid_map[::DS, ::DS]
            inflated_cluttered = binary_dilation(small_grid_cluttered == 1, iterations=small_safe_radius)
            pf_matrix_cluttered = (~inflated_cluttered).astype(int)
            pf_grid_cluttered = Grid(matrix=pf_matrix_cluttered.tolist())

            s_node_c = pf_grid_cluttered.node(rooms[start_idx][0] // DS, rooms[start_idx][1] // DS)
            e_node_c = pf_grid_cluttered.node(rooms[end_idx][0] // DS, rooms[end_idx][1] // DS)

            if not s_node_c.walkable or not e_node_c.walkable: continue
            path_nodes_cluttered, _ = finder.find_path(s_node_c, e_node_c, pf_grid_cluttered)
            if not path_nodes_cluttered: continue

            return grid_map, sparse_waypoints, resolution

    @staticmethod
    def create_fallback_slalom(size_x, size_y, resolution):
            """
            RANDOMIZED SLALOM: A brutal but UNIQUE fallback map.
            Prevents dataset poisoning by ensuring no two fallback maps are identical.
            """
            grid_w = int(size_x * resolution)
            grid_h = int(size_y * resolution)
            grid = np.ones((grid_h, grid_w), dtype=np.int8)
            grid[5:-5, 5:-5] = 0 # Clear the center

            # 1. Randomize the geometry
            num_baffles = random.randint(3, 6) # 3 to 6 turns
            base_baffle_length = grid_w * random.uniform(0.55, 0.70) # 55% to 70% of room width
            gap_y = grid_h // (num_baffles + 1)
            wall_thickness = int(random.uniform(0.4, 0.8) * resolution) 

            for i in range(1, num_baffles + 1):
                # Jitter the Y placement so the gaps aren't perfectly uniform
                y = int(i * gap_y + random.uniform(-1.5, 1.5) * resolution)

                # Ensure Y stays within bounds
                y = max(5, min(grid_h - wall_thickness - 5, y))

                # Jitter the baffle length
                baffle_len = int(base_baffle_length + random.uniform(-2.0, 2.0) * resolution)

                if i % 2 == 0:
                    grid[y:y+wall_thickness, 0:baffle_len] = 1 
                else:
                    grid[y:y+wall_thickness, grid_w-baffle_len:grid_w] = 1 

            # 2. Jitter the Waypoints
            # Ensures the robot doesn't memorize a perfect straight line down the middle
            waypoints = np.array([
                [size_x * random.uniform(0.4, 0.6), size_y * 0.1], 
                [size_x * random.uniform(0.4, 0.6), size_y * 0.5], 
                [size_x * random.uniform(0.4, 0.6), size_y * 0.9]
            ])

            return grid, waypoints, resolution

    @staticmethod
    def preview(difficulty=0.5):
        print(f"Generating Map Preview (Difficulty: {difficulty})...")
        grid, wp, res = MapGenerator.generate(size_x=40, size_y=40, difficulty=difficulty)

        plt.figure(figsize=(8, 8))
        plt.imshow(grid == 0, cmap='gray', origin='lower')

        wp_px = wp * res
        plt.plot(wp_px[:, 0], wp_px[:, 1], 'r--', linewidth=2, label='Path')
        plt.scatter(wp_px[:, 0], wp_px[:, 1], c='red', s=50, label='Waypoints')
        plt.scatter(wp_px[0, 0], wp_px[0, 1], c='lime', s=150, marker='*', label='Start')
        plt.scatter(wp_px[-1, 0], wp_px[-1, 1], c='blue', s=150, marker='X', label='Goal')

        plt.title(f"BEANS Autonomy Map | Difficulty: {difficulty} | Res: {res} px/m")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    while True:
        try:
            val = input("Enter a difficulty between 0.0 and 1.0 (or 'q' to quit): ")
            if val.lower() == 'q':
                break
            diff = float(val)
            MapGenerator.preview(difficulty=diff)
        except ValueError:
            print("Invalid input.")
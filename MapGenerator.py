import numpy as np
import random
import math
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import binary_dilation

class MapGenerator:
    
    # --- C-ACCELERATED VECTORIZED RAYCAST ---
    @staticmethod
    def _is_los_clear_ds(x0, y0, x1, y1, pf_matrix):
        """
        Calculates line of sight using Numpy vectorization (compiled in C).
        pf_matrix: 1 is Walkable, 0 is Wall.
        """
        dist = max(abs(x1 - x0), abs(y1 - y0))
        if dist == 0: 
            return True
            
        # Generates array indices instantly in C
        xs = np.linspace(x0, x1, num=dist, endpoint=True).astype(int)
        ys = np.linspace(y0, y1, num=dist, endpoint=True).astype(int)
        
        # If all pixels on the line equal 1 (Free space), LOS is clear
        return np.all(pf_matrix[ys, xs] == 1)

    @staticmethod
    def generate(size_x=40, size_y=40):
        resolution = 50  
        grid_w = int(size_x * resolution)
        grid_h = int(size_y * resolution)

        corridor_width_m = 1.5 
        corridor_px = int(round(corridor_width_m * resolution))

        room_min_m = 3.0 
        room_max_m = 6.0
        room_min_px = int(round(room_min_m * resolution))
        room_max_px = int(round(room_max_m * resolution))

        clutter_size_m = 0.35 
        clutter_px = int(round(clutter_size_m * resolution))

        safe_radius_m = 0.50 
        safe_radius_px = int(round(safe_radius_m * resolution))

        valid_map = False
        attempts = 0

        while not valid_map:
            attempts += 1
            enable_clutter = (attempts <= 20)

            if attempts > 40:
                return MapGenerator.create_fallback_map(size_x, size_y, resolution)

            grid_map = np.ones((grid_h, grid_w), dtype=np.int8)

            # --- STRUCTURE GENERATION ---
            rooms = []
            sectors_x, sectors_y = 3, 3
            sector_w, sector_h = grid_w // sectors_x, grid_h // sectors_y

            for sx in range(sectors_x):
                for sy in range(sectors_y):
                    if random.random() < 0.8:
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

            for i in range(len(rooms) - 1):
                c1, c2 = rooms[i], rooms[i+1]

                y_start, y_end = c1[1], c1[1] + corridor_px
                x_start, x_end = min(c1[0], c2[0]), max(c1[0], c2[0])
                y_end, x_end = min(y_end, grid_h - 1), min(x_end, grid_w - 1)
                grid_map[y_start:y_end, x_start:x_end] = 0

                if enable_clutter and random.random() > 0.5:
                    ox = random.randint(x_start, x_end - clutter_px) if x_end - clutter_px > x_start else x_start
                    oy = y_start if random.random() > 0.5 else y_end - clutter_px
                    if ox + clutter_px < grid_w and oy + clutter_px < grid_h:
                        clutter_blocks.append((oy, ox)) 

                y_start_v, y_end_v = min(c1[1], c2[1]), max(c1[1], c2[1])
                x_start_v, x_end_v = c2[0], c2[0] + corridor_px
                y_end_v, x_end_v = min(y_end_v, grid_h - 1), min(x_end_v, grid_w - 1)
                grid_map[y_start_v:y_end_v, x_start_v:x_end_v] = 0

                if enable_clutter and random.random() > 0.5:
                    oy = random.randint(y_start_v, y_end_v - clutter_px) if y_end_v - clutter_px > y_start_v else y_start_v
                    ox = x_start_v if random.random() > 0.5 else x_end_v - clutter_px
                    if ox + clutter_px < grid_w and oy + clutter_px < grid_h:
                        clutter_blocks.append((oy, ox)) 

            grid_map[0:2, :] = 1; grid_map[-2:, :] = 1
            grid_map[:, 0:2] = 1; grid_map[:, -2:] = 1

            # --- PRE-DOWNSAMPLING OPTIMIZATION ---
            # Shrink map by 10x FIRST to obliterate CPU load
            DS = 10 
            small_grid = grid_map[::DS, ::DS]
            
            # Calculate the safe radius for the tiny grid
            small_safe_radius = int(math.ceil(safe_radius_px / DS))
            
            # Inflate walls on the tiny grid (Takes 0.001 seconds)
            inflated_small = binary_dilation(small_grid == 1, iterations=small_safe_radius)
            
            # 1 is Walkable, 0 is Wall for pathfinding and raycasting
            pf_matrix = (~inflated_small).astype(int) 
            pf_grid = Grid(matrix=pf_matrix.tolist())

            max_dist = 0
            best_pair = (0, len(rooms)-1)
            for r1 in range(len(rooms)):
                for r2 in range(r1 + 1, len(rooms)):
                    d = math.hypot(rooms[r1][0] - rooms[r2][0], rooms[r1][1] - rooms[r2][1])
                    if d > max_dist:
                        max_dist = d
                        best_pair = (r1, r2)
            
            start_idx, end_idx = best_pair
            
            s_node = pf_grid.node(rooms[start_idx][0] // DS, rooms[start_idx][1] // DS)
            e_node = pf_grid.node(rooms[end_idx][0] // DS, rooms[end_idx][1] // DS)
            
            if not s_node.walkable or not e_node.walkable:
                continue

            # A* now searches a tiny 200x200 grid instead of 2000x2000
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path_nodes, _ = finder.find_path(s_node, e_node, pf_grid)

            if not path_nodes:
                continue

            # --- HIGH-SPEED STRING PULLING ---
            smoothed_nodes = [path_nodes[0]]
            curr_idx = 0
            
            while curr_idx < len(path_nodes) - 1:
                furthest_visible = curr_idx + 1
                for j in range(len(path_nodes) - 1, curr_idx, -1):
                    # Raycast runs on compiled C code now
                    if MapGenerator._is_los_clear_ds(path_nodes[curr_idx].x, path_nodes[curr_idx].y, 
                                                     path_nodes[j].x, path_nodes[j].y, pf_matrix):
                        furthest_visible = j
                        break
                        
                smoothed_nodes.append(path_nodes[furthest_visible])
                curr_idx = furthest_visible
                
            raw_path = []
            for node in smoothed_nodes:
                wx = ((node.x * DS) + (DS / 2.0)) / resolution
                wy = ((node.y * DS) + (DS / 2.0)) / resolution
                raw_path.append([wx, wy])
                
            sparse_waypoints = np.array(raw_path)

            # --- STRICT DISTANCE FILTER (Spatial Downsampling) ---
            final_path = [sparse_waypoints[0]]
            
            for i in range(1, len(sparse_waypoints) - 1):
                p_curr = sparse_waypoints[i]
                # Calculate distance to the LAST mathematically approved point
                dist_to_last = math.hypot(p_curr[0] - final_path[-1][0], p_curr[1] - final_path[-1][1])
                
                # If it's further than 1 meter away, it is a valid, necessary corner.
                if dist_to_last >= 1.5:
                    final_path.append(p_curr)
                    
            # The Final Goal Logic
            final_pt = sparse_waypoints[-1]
            dist_to_goal = math.hypot(final_pt[0] - final_path[-1][0], final_pt[1] - final_path[-1][1])
            
            # If the last corner is awkwardly close to the goal (e.g. 0.2m), 
            # we just overwrite it with the goal to prevent a tiny stutter-step.
            if dist_to_goal < 1.0 and len(final_path) > 1:
                final_path[-1] = final_pt
            elif dist_to_goal >= 0.1:
                final_path.append(final_pt)
                
            sparse_waypoints = np.array(final_path)

            # --- DEFERRED CLUTTER INJECTION ---
            for oy, ox in clutter_blocks:
                grid_map[oy:oy+clutter_px, ox:ox+clutter_px] = 1

            return grid_map, sparse_waypoints, resolution

    @staticmethod
    def create_fallback_map(size_x, size_y, resolution):
        grid_w = int(size_x * resolution)
        grid_h = int(size_y * resolution)
        grid = np.zeros((grid_h, grid_w), dtype=np.int8)
        grid[0:5, :] = 1; grid[-5:, :] = 1
        grid[:, 0:5] = 1; grid[:, -5:] = 1
        start_pt = [size_x * 0.1, size_y * 0.1]
        end_pt = [size_x * 0.9, size_y * 0.9]
        waypoints = np.array([start_pt, end_pt])
        return grid, waypoints, resolution

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("--- Testing High-Speed C-Accelerated Map Generation ---")
    g_map, g_waypoints, res = MapGenerator.generate(40, 40)
    
    plt.figure(figsize=(10, 10))
    # 1 is Wall (Black), 0 is Free (White)
    plt.imshow(1 - g_map, cmap='Greys', origin='lower', extent=[0, 40, 0, 40])
    
    if len(g_waypoints) > 0:
        plt.plot(g_waypoints[:, 0], g_waypoints[:, 1], 'r-o', linewidth=2, label='Global Path (Vectorized)')
        plt.plot(g_waypoints[0, 0], g_waypoints[0, 1], 'bo', markersize=8, label='Start')
        plt.plot(g_waypoints[-1, 0], g_waypoints[-1, 1], 'go', markersize=8, label='End')
    
    plt.legend()
    plt.title(f"Generated Map (Resolution: {res} px/m)")
    plt.show()
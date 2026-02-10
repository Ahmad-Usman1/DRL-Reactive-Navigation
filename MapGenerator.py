import numpy as np
import random
import math
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import binary_dilation  # NEW: For wall inflation

class MapGenerator:
    @staticmethod
    def generate(size_x=20, size_y=20):
        """
        Generates a 2D grid map with safe waypoints.
        Scale: 1.0 unit = 1.0 meter
        Resolution: 50 pixels per meter (2cm per pixel)
        """
        # --- SETTINGS ---
        resolution = 50  # 50 pixels/meter
        grid_w = int(size_x * resolution)
        grid_h = int(size_y * resolution)

        # Dimensions
        corridor_width_m = 2.0
        corridor_px = int(round(corridor_width_m * resolution))

        room_min_m = 3.0
        room_max_m = 6.0
        room_min_px = int(round(room_min_m * resolution))
        room_max_px = int(round(room_max_m * resolution))

        clutter_size_m = 0.5
        clutter_px = int(round(clutter_size_m * resolution))

        # SAFETY MARGIN (New)
        safe_radius_m = 0.35 # 35cm safety bubble
        safe_radius_px = int(round(safe_radius_m * resolution))

        valid_map = False
        attempts = 0

        while not valid_map:
            attempts += 1
            enable_clutter = (attempts <= 20)

            if attempts > 40:
                print("MapGenerator: Complex generation failed. Using fallback.")
                return MapGenerator.create_fallback_map(size_x, size_y, resolution)

            # 1. Init Grid (1=Wall, 0=Free)
            grid_map = np.ones((grid_h, grid_w), dtype=np.int8)

            # 2. Rooms
            num_rooms = random.randint(5, 9)
            rooms = [] 

            for _ in range(num_rooms):
                w = random.randint(room_min_px, room_max_px)
                h = random.randint(room_min_px, room_max_px)
                
                if grid_w - w - 2 <= 2 or grid_h - h - 2 <= 2: continue
                    
                x = random.randint(2, grid_w - w - 2)
                y = random.randint(2, grid_h - h - 2)

                grid_map[y:y+h, x:x+w] = 0
                rooms.append([int(x + w/2), int(y + h/2)])

            if len(rooms) < 2: continue

            # 3. Corridors
            for i in range(len(rooms) - 1):
                c1 = rooms[i]
                c2 = rooms[i+1]

                # Horizontal
                y_start, y_end = c1[1], c1[1] + corridor_px
                x_start, x_end = min(c1[0], c2[0]), max(c1[0], c2[0])
                y_end, x_end = min(y_end, grid_h - 1), min(x_end, grid_w - 1)
                grid_map[y_start:y_end, x_start:x_end] = 0

                # Vertical
                y_start_v, y_end_v = min(c1[1], c2[1]), max(c1[1], c2[1])
                x_start_v, x_end_v = c2[0], c2[0] + corridor_px
                y_end_v, x_end_v = min(y_end_v, grid_h - 1), min(x_end_v, grid_w - 1)
                grid_map[y_start_v:y_end_v, x_start_v:x_end_v] = 0

                # Clutter
                if enable_clutter and random.random() > 0.4:
                    ox = random.randint(x_start, x_end) if x_end > x_start else x_start
                    oy = c1[1] + int(corridor_px/2)
                    if ox + clutter_px < grid_w and oy + clutter_px < grid_h:
                        grid_map[oy:oy+clutter_px, ox:ox+clutter_px] = 1

            # Borders
            grid_map[0:2, :] = 1; grid_map[-2:, :] = 1
            grid_map[:, 0:2] = 1; grid_map[:, -2:] = 1

            # --- 4. SAFE PATHFINDING (Inflation Step) ---
            # Create a "Planning Map" where walls are thicker
            # We treat 1s as walls. dilation expands the 1s.
            inflated_map = binary_dilation(grid_map, iterations=safe_radius_px)
            
            # Pathfinding lib needs 1=Walkable, 0=Wall. 
            # So we invert the INFLATED map.
            pf_matrix = (~inflated_map).astype(int) 
            pf_grid = Grid(matrix=pf_matrix)

            # Pick furthest rooms
            max_dist = 0
            best_pair = (0, len(rooms)-1)
            for r1 in range(len(rooms)):
                for r2 in range(r1 + 1, len(rooms)):
                    d = math.hypot(rooms[r1][0] - rooms[r2][0], rooms[r1][1] - rooms[r2][1])
                    if d > max_dist:
                        max_dist = d
                        best_pair = (r1, r2)
            
            start_idx, end_idx = best_pair
            
            # Verify Start/End are not inside the inflated walls
            # If start room is too small and got closed off by inflation, skip map
            s_node = pf_grid.node(rooms[start_idx][0], rooms[start_idx][1])
            e_node = pf_grid.node(rooms[end_idx][0], rooms[end_idx][1])
            
            if not s_node.walkable or not e_node.walkable:
                continue

            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path_nodes, _ = finder.find_path(s_node, e_node, pf_grid)

            if not path_nodes:
                continue

            # 5. Extract Waypoints
            raw_path = []
            for node in path_nodes:
                # Convert Grid (Px) to World (Meters)
                wx = (node.x + 0.5) / resolution
                wy = (node.y + 0.5) / resolution
                raw_path.append([wx, wy])
            
            raw_path = np.array(raw_path)
            sparse_waypoints = [raw_path[0]]
            
            if len(raw_path) > 2:
                prev_dir = math.atan2(raw_path[1][1] - raw_path[0][1], 
                                      raw_path[1][0] - raw_path[0][0])
                
                for i in range(1, len(raw_path) - 1):
                    curr_dir = math.atan2(raw_path[i+1][1] - raw_path[i][1], 
                                          raw_path[i+1][0] - raw_path[i][0])
                    
                    diff = MapGenerator.angdiff(prev_dir, curr_dir)
                    if abs(diff) > math.radians(45):
                        last_wp = sparse_waypoints[-1]
                        dist = math.hypot(raw_path[i][0] - last_wp[0], 
                                          raw_path[i][1] - last_wp[1])
                        # Ensure waypoints aren't bunched up
                        if dist > 3.0:
                            sparse_waypoints.append(raw_path[i])
                            prev_dir = curr_dir

            final_pt = raw_path[-1]
            if np.linalg.norm(np.array(sparse_waypoints[-1]) - final_pt) > 0.1:
                sparse_waypoints.append(final_pt)

            # Return original map (not inflated) so robot sees real walls
            return grid_map, np.array(sparse_waypoints), resolution

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

    @staticmethod
    def angdiff(th1, th2):
        return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Generating map with safety inflation...")
    g_map, g_waypoints, res = MapGenerator.generate(20, 20)
    
    plt.figure(figsize=(10, 10))
    # '1' is Wall, so we plot it as black (cmap Greys: 0=White, 1=Black usually, depends on range)
    # Actually standard Greys: 0 is white, 1 is black is intuitive.
    plt.imshow(1-g_map, cmap='Greys', origin='lower', extent=[0, 20, 0, 20])
    
    if len(g_waypoints) > 0:
        plt.plot(g_waypoints[:, 0], g_waypoints[:, 1], 'r-o', linewidth=2, label='Safe Path')
        plt.plot(g_waypoints[0, 0], g_waypoints[0, 1], 'bo', label='Start')
        plt.plot(g_waypoints[-1, 0], g_waypoints[-1, 1], 'go', label='End')
    
    plt.legend()
    plt.title(f"Generated Map (Resolution: {res} px/m)")
    plt.show()
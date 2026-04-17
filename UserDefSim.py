import pygame
import numpy as np
import onnxruntime as ort
import math
import heapq
from numba import njit
from scipy.ndimage import binary_dilation

# --- HIGH-PERFORMANCE C-COMPILED RAYCASTER ---
@njit(fastmath=True)
def fast_raycast(rx, ry, rth, angles, map_grid, res_m, max_range):
    num_rays = angles.shape[0] 
    scan = np.zeros(num_rays, dtype=np.float32)
    h, w = map_grid.shape
    step_m = 0.04 
    
    for i in range(num_rays):
        glob_angle = rth + angles[i]
        dx = math.cos(glob_angle) * step_m
        dy = math.sin(glob_angle) * step_m
        
        curr_x = rx
        curr_y = ry
        dist_m = 0.0
        
        while dist_m < max_range:
            dist_m += step_m
            curr_x += dx
            curr_y += dy
            
            ix = int(curr_x / res_m)
            iy = int(curr_y / res_m)
            
            if ix < 0 or ix >= w or iy < 0 or iy >= h: break
            if map_grid[iy, ix] == 1: break
                
        scan[i] = min(max_range, dist_m)
    return scan

# --- CONFIGURATION ---
GRID_W, GRID_H = 40, 20
CELL_SIZE = 20  
RES_M = 0.5     
WIDTH, HEIGHT = GRID_W * CELL_SIZE, GRID_H * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)  
DARK_RED = (150, 0, 0)
DARK_GRAY = (100, 100, 100)
FAINT_LINE = (40, 40, 40)

class BEANSSimulator:
    def __init__(self, onnx_path):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + 100))
        pygame.display.set_caption("BEANS Visibility Graph Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.btn_font = pygame.font.SysFont("Arial", 14, bold=True)

        self.grid = np.zeros((GRID_H, GRID_W), dtype=int)
        
        self.mode = "DRAW_WALL" 
        self.state = "SETUP"
        self.btn_stop = pygame.Rect(WIDTH - 300, HEIGHT + 30, 80, 40)
        self.btn_clear_wp = pygame.Rect(WIDTH - 200, HEIGHT + 30, 90, 40)
        self.btn_reset = pygame.Rect(WIDTH - 90, HEIGHT + 30, 80, 40)
        
        self.start_pos = None 
        self.waypoints = []   
        self.vis_graph = {} 
        self.active_goal = None
        self.global_path = [] 
        self.driven_trajectory = [] 
        
        # Navigation Tuning
        self.is_aligning = False
        self.align_threshold = 0.15 
        self.align_omega = 1.0      
        self.capture_radius = 1.2 # MASSIVELY EXPANDED to prevent Wall-Repulsion loops
        
        # Hardware Specs
        self.wheel_radius = 0.0955  
        self.wheel_base = 0.33      
        self.robot_radius = 0.31  
        self.max_lin_vel = 0.4 
        self.max_ang_vel = 2.0 
        self.max_wheel_omega = (self.max_lin_vel + (self.max_ang_vel * self.wheel_base / 2.0)) / self.wheel_radius
        self.tau_motor = 0.30 
        
        self.max_sensor_range = 5.0 
        self.dt = 0.1
        self.lag_steps = 3 
        
        front_degs = [90, 50, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -50, -90]
        self.sensor_angles = np.deg2rad(front_degs).astype(np.float32)
        
        self.pose = [0.0, 0.0, 0.0]
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        
        print(f"Loading ONNX engine from {onnx_path}...")
        self.session = ort.InferenceSession(onnx_path)

    def _px_to_m(self, px, py): return (px / CELL_SIZE) * RES_M, (py / CELL_SIZE) * RES_M
    def _m_to_px(self, mx, my): return int((mx / RES_M) * CELL_SIZE), int((my / RES_M) * CELL_SIZE)
    def _angdiff(self, th1, th2): return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi

    def _check_rigid_collision(self, x, y):
        for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            px = x + self.robot_radius * math.cos(ang)
            py = y + self.robot_radius * math.sin(ang)
            ix, iy = int(px / RES_M), int(py / RES_M)
            if ix < 0 or ix >= GRID_W or iy < 0 or iy >= GRID_H: return True
            if self.grid[iy, ix] == 1: return True
        return False

    def _check_line_of_sight(self, p1, p2, inflated_grid):
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = int(dist / 0.1) 
        if steps == 0: return True
        
        for i in range(steps + 1):
            t = i / steps
            cx = p1[0] + t * (p2[0] - p1[0])
            cy = p1[1] + t * (p2[1] - p1[1])
            ix, iy = int(cx / RES_M), int(cy / RES_M)
            if ix < 0 or ix >= GRID_W or iy < 0 or iy >= GRID_H: return False
            if inflated_grid[iy, ix] == 1: return False
        return True

    def update_visibility_graph(self):
        inflated = binary_dilation(self.grid == 1, iterations=1)
        self.vis_graph = {i: [] for i in range(len(self.waypoints))}
        
        for i in range(len(self.waypoints)):
            for j in range(i + 1, len(self.waypoints)):
                if self._check_line_of_sight(self.waypoints[i], self.waypoints[j], inflated):
                    dist = math.hypot(self.waypoints[i][0] - self.waypoints[j][0], 
                                      self.waypoints[i][1] - self.waypoints[j][1])
                    self.vis_graph[i].append((j, dist))
                    self.vis_graph[j].append((i, dist))

    def plan_waypoint_path(self, start_m, goal_idx):
        inflated = binary_dilation(self.grid == 1, iterations=1)
        
        start_connections = []
        for i, wp in enumerate(self.waypoints):
            if self._check_line_of_sight(start_m, wp, inflated):
                dist = math.hypot(start_m[0] - wp[0], start_m[1] - wp[1])
                start_connections.append((i, dist))
                
        if not start_connections: return None 
        
        queue = [(0.0, -1, [])] 
        visited = set()
        
        while queue:
            cost, curr, path = heapq.heappop(queue)
            
            if curr == goal_idx:
                return [self.waypoints[i] for i in path]
                
            if curr in visited: continue
            visited.add(curr)
            
            neighbors = start_connections if curr == -1 else self.vis_graph[curr]
            
            for neighbor, dist in neighbors:
                if neighbor not in visited:
                    # FIX 1: Non-Linear penalty (dist ** 1.2) forces Dijkstra to use intermediate stepping stones
                    penalty_weight = dist ** 1.2 
                    heapq.heappush(queue, (cost + penalty_weight, neighbor, path + [neighbor]))
                    
        return None 

    def draw_ui(self):
        self.screen.fill(BLACK)
        
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = GRAY if self.grid[y, x] == 1 else WHITE
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        for node, neighbors in self.vis_graph.items():
            p1 = self._m_to_px(self.waypoints[node][0], self.waypoints[node][1])
            for neighbor, _ in neighbors:
                p2 = self._m_to_px(self.waypoints[neighbor][0], self.waypoints[neighbor][1])
                pygame.draw.line(self.screen, FAINT_LINE, p1, p2, 1)

        for wp in self.waypoints:
            px, py = self._m_to_px(wp[0], wp[1])
            color = RED if wp == self.active_goal else YELLOW
            pygame.draw.circle(self.screen, color, (px, py), 8)
            
        if self.start_pos:
            px, py = self._m_to_px(self.start_pos[0], self.start_pos[1])
            pygame.draw.circle(self.screen, GREEN, (px, py), 10)

        if len(self.driven_trajectory) > 1:
            traj_pts = [self._m_to_px(p[0], p[1]) for p in self.driven_trajectory]
            pygame.draw.lines(self.screen, CYAN, False, traj_pts, 4)

        if self.global_path:
            pts = [self._m_to_px(self.pose[0], self.pose[1])] + [self._m_to_px(p[0], p[1]) for p in self.global_path]
            if len(pts) > 1: pygame.draw.lines(self.screen, RED, False, pts, 3)
            
        if self.state == "RUNNING":
            px, py = self._m_to_px(self.pose[0], self.pose[1])
            pygame.draw.circle(self.screen, BLUE, (px, py), int((self.robot_radius/RES_M)*CELL_SIZE))
            hx = px + int(math.cos(self.pose[2]) * 20)
            hy = py + int(math.sin(self.pose[2]) * 20)
            pygame.draw.line(self.screen, MAGENTA, (px, py), (hx, hy), 3)

        ui_bg = pygame.Rect(0, HEIGHT, WIDTH, 100)
        pygame.draw.rect(self.screen, (30, 30, 30), ui_bg)
        
        controls = "Keys: [W] Wall | [P] Waypoint | [E] Erase | Hover+[S] Start | [ENTER] Run"
        status_text = "ALIGNING..." if self.is_aligning else "ONNX DRIVING"
        status = f"State: {self.state} | Tool: {self.mode} | AI: {status_text if self.active_goal else 'IDLE'}"
        
        t1 = self.font.render(controls, True, WHITE)
        t2 = self.font.render(status, True, YELLOW if self.active_goal else WHITE)
        self.screen.blit(t1, (10, HEIGHT + 15))
        self.screen.blit(t2, (10, HEIGHT + 45))
        
        pygame.draw.rect(self.screen, DARK_RED, self.btn_stop)
        pygame.draw.rect(self.screen, WHITE, self.btn_stop, 2)
        self.screen.blit(self.btn_font.render("STOP AI", True, WHITE), (self.btn_stop.x + 12, self.btn_stop.y + 12))

        pygame.draw.rect(self.screen, DARK_GRAY, self.btn_clear_wp)
        pygame.draw.rect(self.screen, WHITE, self.btn_clear_wp, 2)
        self.screen.blit(self.btn_font.render("CLEAR WPs", True, WHITE), (self.btn_clear_wp.x + 8, self.btn_clear_wp.y + 12))

        pygame.draw.rect(self.screen, DARK_RED, self.btn_reset)
        pygame.draw.rect(self.screen, WHITE, self.btn_reset, 2)
        self.screen.blit(self.btn_font.render("RESET MAP", True, WHITE), (self.btn_reset.x + 5, self.btn_reset.y + 12))

        pygame.display.flip()

    def handle_click(self, pos, button):
        if self.btn_stop.collidepoint(pos):
            self.active_goal = None
            self.global_path = []
            self.current_lin_vel = self.current_ang_vel = 0.0
            self.action_history.fill(0.0)
            self.is_aligning = False
            return
            
        if self.btn_clear_wp.collidepoint(pos):
            self.waypoints = []
            self.vis_graph = {}
            self.active_goal = None
            self.global_path = []
            self.driven_trajectory = []
            return

        if self.btn_reset.collidepoint(pos):
            self.grid.fill(0)
            self.waypoints = []
            self.vis_graph = {}
            self.start_pos = None
            self.active_goal = None
            self.global_path = []
            self.driven_trajectory = []
            self.state = "SETUP"
            return

        if pos[1] >= HEIGHT: return 
        mx, my = self._px_to_m(pos[0], pos[1])
        gx, gy = int(pos[0] // CELL_SIZE), int(pos[1] // CELL_SIZE)

        clicked_wp_idx = -1
        for i, wp in enumerate(self.waypoints):
            if math.hypot(wp[0] - mx, wp[1] - my) < 0.6: 
                clicked_wp_idx = i
                break

        if button == 3: 
            if clicked_wp_idx != -1: self.waypoints.pop(clicked_wp_idx)
            else: self.grid[gy, gx] = 0
            self.update_visibility_graph()
            return

        if button == 1: 
            if self.state == "SETUP":
                if self.mode == "DRAW_WALL":
                    self.grid[gy, gx] = 1
                    self.update_visibility_graph()
                elif self.mode == "PLACE_WP":
                    if self.grid[gy, gx] == 0: 
                        self.waypoints.append((mx, my))
                        self.update_visibility_graph()
                elif self.mode == "ERASE_WALL":
                    self.grid[gy, gx] = 0
                    self.update_visibility_graph()
                    
            elif self.state == "RUNNING":
                if clicked_wp_idx != -1:
                    path = self.plan_waypoint_path((self.pose[0], self.pose[1]), clicked_wp_idx)
                    if path:
                        self.global_path = path
                        self.active_goal = self.waypoints[clicked_wp_idx]
                        self.is_aligning = True 
                        self.driven_trajectory = [(self.pose[0], self.pose[1])]
                else:
                    if self.grid[gy, gx] == 0:
                        self.waypoints.append((mx, my))
                        self.update_visibility_graph()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w: self.mode = "DRAW_WALL"
                    if event.key == pygame.K_p: self.mode = "PLACE_WP"
                    if event.key == pygame.K_e: self.mode = "ERASE_WALL"
                    if event.key == pygame.K_s:
                        mx, my = self._px_to_m(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
                        self.start_pos = (mx, my)
                        self.pose = [mx, my, 0.0]
                    if event.key == pygame.K_RETURN and self.start_pos:
                        self.state = "RUNNING"
                        self.pose = [self.start_pos[0], self.start_pos[1], 0.0]
                        self.driven_trajectory = []

                if pygame.mouse.get_pressed()[0]: self.handle_click(pygame.mouse.get_pos(), 1)
                elif pygame.mouse.get_pressed()[2]: self.handle_click(pygame.mouse.get_pos(), 3)

            if self.state == "RUNNING" and self.active_goal and self.global_path:
                
                local_goal = self.global_path[0]
                dist_to_local = math.hypot(local_goal[0] - self.pose[0], local_goal[1] - self.pose[1])
                
                # FIX 2: Expanded Capture Radius (1.2m)
                if dist_to_local < self.capture_radius and len(self.global_path) > 1:
                    self.global_path.pop(0)
                    local_goal = self.global_path[0]
                    
                    # Re-align if the new segment requires a sharp turn (>45 degrees)
                    new_desired_angle = math.atan2(local_goal[1] - self.pose[1], local_goal[0] - self.pose[0])
                    if abs(self._angdiff(self.pose[2], new_desired_angle)) > math.radians(45):
                        self.is_aligning = True 

                dist_to_final = math.hypot(self.active_goal[0] - self.pose[0], self.active_goal[1] - self.pose[1])
                if dist_to_final < 0.5:
                    self.active_goal = None
                    self.global_path = []
                    self.current_lin_vel = self.current_ang_vel = 0.0
                    self.action_history.fill(0.0)
                    continue

                desired_angle = math.atan2(local_goal[1] - self.pose[1], local_goal[0] - self.pose[0])
                heading_err = self._angdiff(self.pose[2], desired_angle)
                
                if self.is_aligning:
                    if abs(heading_err) > self.align_threshold:
                        target_v = 0.0
                        target_w = np.sign(heading_err) * self.align_omega
                        self.action_history[:-1] = self.action_history[1:]
                        self.action_history[-1] = [0.0, target_w / self.max_ang_vel]
                    else:
                        self.is_aligning = False
                        target_v = self.current_lin_vel
                        target_w = self.current_ang_vel
                else:
                    sonars = fast_raycast(self.pose[0], self.pose[1], self.pose[2],
                                          self.sensor_angles, self.grid, RES_M, self.max_sensor_range)
                    
                    norm_scan = np.clip(sonars, 0, self.max_sensor_range) / self.max_sensor_range
                    norm_dist = min(dist_to_local / 10.0, 1.0)
                    norm_head = heading_err / np.pi
                    norm_v = self.current_lin_vel / self.max_lin_vel
                    norm_w = self.current_ang_vel / self.max_ang_vel
                    
                    obs = np.concatenate([
                        norm_scan, [norm_dist], [norm_head], [norm_v], [norm_w], self.action_history.flatten()
                    ]).astype(np.float32).reshape(1, -1)

                    action = self.session.run(None, {"observation": obs})[0][0]

                    delayed_action = self.action_history[0].copy()
                    self.action_history[:-1] = self.action_history[1:]
                    self.action_history[-1] = action.copy()

                    norm_lin = np.clip(delayed_action[0], -1.0, 1.0)
                    norm_ang = np.clip(delayed_action[1], -1.0, 1.0)
                    target_v = norm_lin * self.max_lin_vel if norm_lin >= 0 else 0.0 
                    target_w = norm_ang * self.max_ang_vel

                req_wl = (target_v - (target_w * self.wheel_base / 2.0)) / self.wheel_radius
                req_wr = (target_v + (target_w * self.wheel_base / 2.0)) / self.wheel_radius
                
                max_req_omega = max(abs(req_wl), abs(req_wr))
                if max_req_omega > self.max_wheel_omega:
                    scale = self.max_wheel_omega / max_req_omega
                    req_wl *= scale; req_wr *= scale
                    
                kinematic_v = (self.wheel_radius / 2.0) * (req_wl + req_wr)
                kinematic_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)
                
                alpha = self.dt / (self.tau_motor + self.dt)
                self.current_lin_vel += alpha * (kinematic_v - self.current_lin_vel)
                self.current_ang_vel += alpha * (kinematic_w - self.current_ang_vel)

                new_x = self.pose[0] + self.current_lin_vel * math.cos(self.pose[2]) * self.dt
                new_y = self.pose[1] + self.current_lin_vel * math.sin(self.pose[2]) * self.dt
                
                if not self._check_rigid_collision(new_x, new_y):
                    self.pose[0] = new_x
                    self.pose[1] = new_y
                else:
                    self.current_lin_vel = 0.0 
                    
                self.pose[2] = self._angdiff(0, self.pose[2] + self.current_ang_vel * self.dt)
                
                if not self.driven_trajectory or math.hypot(self.pose[0] - self.driven_trajectory[-1][0], self.pose[1] - self.driven_trajectory[-1][1]) > 0.05:
                    self.driven_trajectory.append((self.pose[0], self.pose[1]))

            self.draw_ui()
            self.clock.tick(10) 

        pygame.quit()

if __name__ == "__main__":
    sim = BEANSSimulator("Performing_Models/beans_verified.onnx")
    sim.run()
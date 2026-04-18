import cv2
import numpy as np
import time
import socket
import struct
import threading
import onnxruntime as ort
import os
import json

# ================= CONFIGURATION =================
# Network Settings
TCP_HOST = '0.0.0.0'         # Listens for ESP-Cam JPEGs
TCP_PORT = 8080              
UDP_HOST = '0.0.0.0'         # Listens for Main ESP Sonar Telemetry
UDP_PORT = 5000              

MODEL_PATH = "model.onnx"

# --- CAMERA & PPO CONSTANTS ---
IMAGE_WIDTH = 320        # Full native width to preserve 62-degree FOV
IMAGE_HEIGHT = 240       
FLOOR_CROP_ROW = 160     # Keep rows 0 to 160. Discard the floor (160 to 240)
CAMERA_FOV = 62.0        
PIXELS_PER_DEGREE = IMAGE_WIDTH / CAMERA_FOV

# PPO Required Angles (Matches manual: Negative = Left, Positive = Right)
PPO_ANGLES = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
# =================================================

# Global Variables & Threading Locks
current_frame = None
latest_sonars = np.zeros(16) # We will only use indices 2, 3, 4, 5
frame_lock = threading.Lock()
udp_lock = threading.Lock()
new_frame_event = threading.Event()
running = True
net_fps = 0
ai_fps = 0
# --- FUSION VARIABLES ---
global_k_ema = 150000.0  # A starting guess for the scale multiplier
ALPHA = 0.2              # Smoothing factor (Lower = smoother but slower to adapt)

# --- THREAD 1: TCP IMAGE RECEIVER ---
def receive_tcp_stream():
    global current_frame, running, net_fps
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((TCP_HOST, TCP_PORT))
        server_socket.listen(1)
        print(f"TCP: Listening on Port {TCP_PORT} for ESP-Cam...")
    except Exception as e:
        print(f"CRITICAL: Failed to bind TCP socket: {e}")
        running = False
        return

    while running:
        server_socket.settimeout(1.0) 
        try:
            conn, addr = server_socket.accept()
        except socket.timeout:
            continue
        except Exception as e:
            break
            
        print(f"TCP: Camera Connected from {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        frame_count = 0
        last_time = time.time()

        try:
            while running:
                header = conn.recv(4)
                if not header or len(header) < 4:
                    print("TCP: Connection dropped. Waiting for reconnect...")
                    break
                    
                image_len = struct.unpack('<I', header)[0]
                
                if image_len == 0 or image_len > 500000:
                    continue

                image_data = b''
                while len(image_data) < image_len:
                    chunk = conn.recv(image_len - len(image_data))
                    if not chunk: break
                    image_data += chunk
                    
                if len(image_data) != image_len:
                    print("TCP Warning: Incomplete frame received. Dropping.")
                    break 

                img_np = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if frame is not None:
                    with frame_lock:
                        current_frame = frame
                    
                    # WAKE UP THE AI THREAD
                    new_frame_event.set()
                    
                    frame_count += 1
                    if time.time() - last_time >= 1.0:
                        net_fps = frame_count
                        frame_count = 0
                        last_time = time.time()
                        
        except Exception as e:
            print(f"TCP error: {e}")
        finally:
            conn.close()
            
    server_socket.close()

# --- THREAD 2: UDP SONAR RECEIVER ---
def receive_udp_telemetry():
    global latest_sonars, running
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    print(f"UDP: Listening on Port {UDP_PORT} for Sonar Telemetry...")
    
    while running:
        try:
            sock.settimeout(1.0)
            data, _ = sock.recvfrom(1024)
            payload = data.decode('utf-8')
            parsed_data = json.loads(payload)
            
            with udp_lock:
                latest_sonars = np.array(parsed_data['s'])
        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP Error: {e}")
            
    sock.close()

# --- HELPER: EXTRACT PPO ARRAY ---
def get_ppo_array_from_depth(depth_map):
    cropped_depth = depth_map[0:FLOOR_CROP_ROW, :]
    raw_1d_scan = np.max(cropped_depth, axis=0)
    
    ppo_array = []
    center_pixel = IMAGE_WIDTH / 2.0
    
    for angle in PPO_ANGLES:
        pixel_idx = int(center_pixel + (angle * PIXELS_PER_DEGREE))
        pixel_idx = max(0, min(IMAGE_WIDTH - 1, pixel_idx)) # Clamp to bounds
        ppo_array.append(raw_1d_scan[pixel_idx])
        
    return ppo_array

# --- HELPER: FUSE CAMERA AND SONAR ---
def fuse_to_metric(ppo_intensities, sonars):
    global global_k_ema
    metric_array = np.zeros(13)
    
    valid_ks = []
    # Map the PPO Array Indices to the Physical Sonar Indices
    anchors = {0: 5, 4: 4, 8: 3, 12: 2}

    # 1. Calculate a stable Global Scale Factor
    for ppo_idx, sonar_idx in anchors.items():
        s_val = float(sonars[sonar_idx])
        i_val = max(1.0, float(ppo_intensities[ppo_idx]))

        # ONLY use this sonar to calibrate the camera IF it sees a real object
        # Ignore dead sonars (0) and sonars hitting the max range (>2400)
        if 100 < s_val < 2400:
            valid_ks.append(s_val * i_val)

    # 2. Smooth the Scale Factor to eliminate neural network flicker
    if len(valid_ks) > 0:
        current_k = sum(valid_ks) / len(valid_ks)
        # Exponential Moving Average: 20% new data, 80% old data
        global_k_ema = (ALPHA * current_k) + ((1 - ALPHA) * global_k_ema)

    # 3. Apply the stable global scale to all 13 camera points
    for idx in range(13):
        i_val = max(1.0, float(ppo_intensities[idx]))
        metric_array[idx] = global_k_ema / i_val

    # 4. The Paranoia Override (Min-Fusion)
    # Never ignore a physical sonar if it sees something closer than the AI thinks
    for ppo_idx, sonar_idx in anchors.items():
        s_val = float(sonars[sonar_idx])
        if 0 < s_val < 2500: # If the sonar has a valid hit
            metric_array[ppo_idx] = min(metric_array[ppo_idx], s_val)
            
    return np.clip(metric_array, 0, 2500)

# =====================================================================
# MAIN THREAD: ONNX INFERENCE & FUSION
# =====================================================================
print(f"Loading ONNX Model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("CRITICAL ERROR: model.onnx not found!")
    exit()

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype('float32')
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype('float32')

print("Starting Engine...")

# Start Background Receivers
threading.Thread(target=receive_tcp_stream, daemon=True).start()
threading.Thread(target=receive_udp_telemetry, daemon=True).start()

print("Waiting for data streams...")
while current_frame is None and running:
    time.sleep(0.1)

frame_count = 0
last_time = time.time()

while running:
    # Wait strictly for a new image to avoid redundant GPU processing
    if new_frame_event.wait(timeout=1.0):
        new_frame_event.clear()

        with frame_lock:
            frame_copy = current_frame.copy()
            
        # Pre-processing (Squish 320x240 to 256x256)
        img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        img_input = img_input.astype('float32') / 255.0
        img_input = (img_input - mean) / std
        img_input = img_input.transpose(2, 0, 1) 
        img_input = img_input[np.newaxis, :, :, :] 

        # ONNX Inference
        outputs = session.run(None, {input_name: img_input})
        prediction = outputs[0]

        # Post-processing (Stretch 256x256 back to 320x240)
        prediction = cv2.resize(prediction[0], (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        
        depth_min = prediction.min()
        depth_max = prediction.max()
        
        if depth_max - depth_min > 1e-6:
            depth_norm = (prediction - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(prediction)
            
        depth_norm_uint8 = (depth_norm * 255).astype(np.uint8)
        
        # --- PERCEPTION FUSION ---
        # 1. Extract the 13 relative intensities from the depth map
        ppo_intensities = get_ppo_array_from_depth(depth_norm_uint8)
        
        # 2. Grab the freshest UDP sonar data
        with udp_lock:
            current_sonars = latest_sonars.copy()
            
        # 3. Fuse to absolute millimeters
        final_metric_distances = fuse_to_metric(ppo_intensities, current_sonars)
        
        print(f"PPO Distances (mm): {[int(x) for x in final_metric_distances]}")
        # -------------------------

        # Calculate Synchronized AI FPS
        frame_count += 1
        if time.time() - last_time >= 1.0:
            ai_fps = frame_count
            frame_count = 0
            last_time = time.time()

        # Visualization
        depth_color = cv2.applyColorMap(depth_norm_uint8, cv2.COLORMAP_MAGMA)
        combined = np.hstack((frame_copy, depth_color))
        
        cv2.putText(combined, f"NET FPS: {net_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"AI FPS:  {ai_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("ONNX Vision & Depth Map", combined)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
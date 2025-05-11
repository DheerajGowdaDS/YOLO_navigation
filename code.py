import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_large")

# Load YOLOv10 model
yolo_model = YOLO('test.pt')
yolo_model.conf = 0.4

# Initialize DPT depth model
depth_model.eval()

# Define transforms for DPT
def depth_transform(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))  # DPT input size
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = (img - 0.5) / 0.5  # DPT normalization
    return img.unsqueeze(0)

# Camera parameters
focal_length = 500.0
cx, cy = 320, 240
camera_matrix = np.array([[focal_length, 0, cx],
                         [0, focal_length, cy],
                         [0, 0, 1]])

# Grid map parameters
map_size = (100, 100)
meters_per_cell = 0.1
map_center = (map_size[1]//2, map_size[0]//2)
grid_map = np.zeros(map_size, dtype=np.uint8)

def pixel_to_world(u, v, depth):
    x = (u - cx) * depth / focal_length
    z = (v - cy) * depth / focal_length
    return x, 0, z  # Assuming flat ground (ignore Y-axis)

def update_map(x, z):
    gx = int(map_center[0] + x/meters_per_cell)
    gy = int(map_center[1] - z/meters_per_cell)
    if 0 <= gx < map_size[1] and 0 <= gy < map_size[0]:
        grid_map[gy, gx] = 1

# Video processing
cap = cv2.VideoCapture("vid.mp4")

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # YOLOv10 detection
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Depth estimation
        input_tensor = depth_transform(frame)
        depth_pred = depth_model(input_tensor).squeeze()
        depth_map = cv2.resize(depth_pred.cpu().numpy(),
                              (frame.shape[1], frame.shape[0]))

        # Process detections
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            u = int((x1 + x2)/2)
            v = int((y1 + y2)/2)
            depth = depth_map[v, u]
            x, _, z = pixel_to_world(u, v, depth)
            update_map(x, z)

        # Visualize
        cv2.imshow("Map", cv2.resize(grid_map*255, (400,400),
                  interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
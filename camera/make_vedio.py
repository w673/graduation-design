import cv2
import os

# ===== 路径（与采集部分完全一致）=====
rgb_dir = "rgb_frames"
depth_dir = "depth_frames"
timestamp_path = "timestamps.txt"

# ===== 读取时间戳 =====
timestamps = []
frame_ids = []

with open(timestamp_path, "r") as f:
    for line in f:
        idx, t = line.strip().split()
        frame_ids.append(int(idx))
        timestamps.append(float(t))

# ===== 排序（确保顺序一致）=====
frame_ids_sorted = sorted(frame_ids)

# ===== 计算真实 FPS =====
fps_list = []
for i in range(1, len(timestamps)):
    dt = timestamps[i] - timestamps[i - 1]
    if dt > 0:
        fps_list.append(1.0 / dt)

fps = sum(fps_list) / len(fps_list)
print(f"Real FPS: {fps:.2f}")

# ===== 读取第一帧（确定尺寸）=====
first_rgb_path = os.path.join(rgb_dir, f"{frame_ids_sorted[0]:06d}.png")
first_depth_path = os.path.join(depth_dir, f"{frame_ids_sorted[0]:06d}.png")

rgb_img = cv2.imread(first_rgb_path)
depth_img = cv2.imread(first_depth_path, cv2.IMREAD_GRAYSCALE)

h_rgb, w_rgb = rgb_img.shape[:2]
h_d, w_d = depth_img.shape[:2]

# ===== VideoWriter =====
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

rgb_writer = cv2.VideoWriter("rgb_strict.mp4", fourcc, fps, (w_rgb, h_rgb))
depth_writer = cv2.VideoWriter("depth_strict.mp4", fourcc, fps, (w_d, h_d), False)

print("Writing videos...")

# ===== 写入（严格按frame_id顺序）=====
for idx in frame_ids_sorted:
    rgb_path = os.path.join(rgb_dir, f"{idx:06d}.png")
    depth_path = os.path.join(depth_dir, f"{idx:06d}.png")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Skip missing frame {idx}")
        continue

    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    rgb_writer.write(rgb)
    depth_writer.write(depth)

rgb_writer.release()
depth_writer.release()

print("Saved rgb_strict.mp4 and depth_strict.mp4")
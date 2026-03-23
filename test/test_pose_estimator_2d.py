import sys
sys.path.append('/data1/fwl_files/design')

import os
import cv2
from gait.pose_estimator_2d import PoseEstimator2D

# =========================
# 输入输出路径
# =========================
rgb_dir = "/data1/fwl_files/rgb_frames"         # 连续帧图片目录
timestamp_path = "/data1/fwl_files/timestamps.txt"  # 时间戳文件
output_video = "/data1/fwl_files/design/result/output_2d_pose_mediapipe.mp4"

# =========================
# 读取时间戳
# =========================
frame_ids = []
timestamps = []

with open(timestamp_path) as f:
    for line in f:
        idx, t = line.strip().split()
        frame_ids.append(int(idx))
        timestamps.append(float(t))

frame_ids = sorted(frame_ids)
timestamps = [timestamps[i] for i in range(len(timestamps))]  # 对应帧顺序

# =========================
# 初始化
# =========================
estimator = PoseEstimator2D()

# 根据时间戳计算真实平均 FPS
fps_list = []
for i in range(1, len(timestamps)):
    dt = timestamps[i] - timestamps[i-1]
    if dt > 0:
        fps_list.append(1.0 / dt)

fps = sum(fps_list) / len(fps_list)
print(f"Estimated FPS from timestamps: {fps:.2f}")

# 读取第一帧确定视频尺寸
first_frame_path = os.path.join(rgb_dir, f"{frame_ids[0]:06d}.png")
first_frame = cv2.imread(first_frame_path)
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# =========================
# 主循环
# =========================
for i, idx in enumerate(frame_ids):

    frame_path = os.path.join(rgb_dir, f"{idx:06d}.png")
    if not os.path.exists(frame_path):
        continue

    frame = cv2.imread(frame_path)

    skeleton_2d, _ = estimator.detect(frame)

    if skeleton_2d is not None:
        vis = estimator.draw_skeleton(frame, skeleton_2d)
    else:
        vis = frame

    writer.write(vis)

    if i % 50 == 0:
        print(f"Processed {i} frames...")

# =========================
# 释放资源
# =========================
writer.release()
print(f"Saved result video to: {output_video}")
import sys
sys.path.append('/data1/fwl_files/design')
import os
import cv2
import numpy as np

from gait.pose_estimator_2d import PoseEstimator2D
from gait.pose_reconstructor_3d import PoseReconstructor3D, CameraParam

# =========================
# 路径
# =========================
rgb_dir = "/data1/fwl_files/middle_demo_2/rgb_frames"
depth_dir = "/data1/fwl_files/middle_demo_2/depth_frames"
timestamp_path = "/data1/fwl_files/middle_demo_2/timestamps.txt"

# =========================
# 初始化
# =========================
pose2d = PoseEstimator2D()
cam = CameraParam()
reconstructor = PoseReconstructor3D(cam, pose2d)

# =========================
# 帧列表
# =========================
frame_ids = []

with open(timestamp_path) as f:
    for line in f:
        idx, _ = line.split()
        frame_ids.append(int(idx))

frame_ids = sorted(frame_ids)

# =========================
# 主循环
# =========================
for idx in frame_ids:

    rgb_path = os.path.join(rgb_dir, f"{idx:06d}.png")
    depth_path = os.path.join(depth_dir, f"{idx:06d}.npy")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        continue

    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path)

    skeleton_3d, skeleton_pit = reconstructor.reconstruct(rgb, depth)

    if skeleton_3d is None:
        continue
    reconstructor.save_3d_plot(skeleton_3d)
# =========================
# 保存结果 + 生成视频
# =========================
reconstructor.save_all()
reconstructor.make_video(timestamp_path, "3d.mp4")

print("All done!")
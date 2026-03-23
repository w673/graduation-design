import sys
sys.path.append('/data1/fwl_files/design')

import cv2
from gait.pose_estimator_2d_mediapipe import PoseEstimator2D


# -----------------------------
# 输入输出路径
# -----------------------------
input_path = "/data1/fwl_files/rgb_strict.mp4"
output_path = "/data1/fwl_files/design/result/output_2d_pose_mediapipe.mp4"


# -----------------------------
# 初始化
# -----------------------------
cap = cv2.VideoCapture(input_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

estimator = PoseEstimator2D()


# -----------------------------
# 主循环
# -----------------------------
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    skeleton = estimator.detect(frame)

    if skeleton is not None:
        vis = estimator.draw_skeleton(frame, skeleton)
    else:
        vis = frame

    writer.write(vis)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames...")


# -----------------------------
# 释放资源
# -----------------------------
cap.release()
writer.release()

print(f"Saved result to: {output_path}")
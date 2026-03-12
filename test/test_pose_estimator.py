import os
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gait.pose_estimator import PoseEstimator, draw_skeleton

# -----------------------------
# 模型配置（写死）
# -----------------------------
sam_checkpoint = "/data1/fwl_files/design/model/sam/sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
device = "cuda:4"


det_config: str = "/data1/fwl_files/gait-analysis/mmpose-main/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
det_checkpoint: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
pose_config: str = "/data1/fwl_files/gait-analysis/mmpose-main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
pose_checkpoint: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"

# -----------------------------
# 初始化 PoseEstimator
# -----------------------------
pose_estimator = PoseEstimator(
    pose_config=pose_config,
    pose_checkpoint=pose_checkpoint,
    mmdet_config=det_config,
    mmdet_checkpoint=det_checkpoint,
    sam_checkpoint=sam_checkpoint,
    device=device
)

# -----------------------------
# 输入视频
# -----------------------------
video_path = "/data1/fwl_files/design/data/test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("无法打开视频")

# 视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 输出视频
output_path = "/data1/fwl_files/design/result/test_pose_out.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 保存为 mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------
# 主循环
# -----------------------------
for _ in range(frame_count):

    ret, frame = cap.read()
    if not ret:
        break

    # pose 估计
    skeleton = pose_estimator.detect(frame)

    if skeleton is not None:
        # 可视化
        vis = draw_skeleton(frame, skeleton)
    else:
        vis = frame

    # 写入输出视频
    out.write(vis)

cap.release()
out.release()

print(f"结果视频已保存到: {output_path}")
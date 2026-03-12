# test_pose_estimator_3d.py
import cv2
import glob
from gait.pose_estimator import PoseEstimator
from gait.pose_estimator_3d import PoseEstimator3D

# -----------------------------
# 初始化2D PoseEstimator
# -----------------------------
pose2d = PoseEstimator(
    pose_config='configs/pose_cfg.py',
    pose_checkpoint='checkpoints/pose.pth',
    mmdet_config='configs/mmdet_cfg.py',
    mmdet_checkpoint='checkpoints/mmdet.pth',
    sam_checkpoint='checkpoints/sam.pth',
    device='cuda:0'
)

# -----------------------------
# 初始化3D PoseEstimator
# -----------------------------
pose3d = PoseEstimator3D(pose2d, depth_scale=0.001)  # 假设深度图单位为mm

# 如果有相机内参，可设置
# pose3d.set_camera_intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

# -----------------------------
# 加载数据
# -----------------------------
rgb_files = sorted(glob.glob('data/rgb/*.png'))  # 或 *.jpg
depth_files = sorted(glob.glob('data/depth/*.png'))

images = [cv2.imread(f) for f in rgb_files]
depth_maps = [cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(float) for f in depth_files]

# -----------------------------
# 批量生成3D骨架
# -----------------------------
skeletons3d = pose3d.detect_batch_3d(images, depth_maps)

# -----------------------------
# 保存3D动画
# -----------------------------
pose3d.save_3d_animation(skeletons3d, out_file='output/skeleton3d.mp4', interval=50)
print("3D骨架动画已保存到 output/skeleton3d.mp4")
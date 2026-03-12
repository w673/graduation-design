import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无窗口服务器可用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

import torch
from gait_transformer_model import GaitTransformer
from gait_analysis import GaitAnalyzer
from pose_estimator_3d import PoseEstimator3D, PoseEstimator  # 你原来的类

# ------------------------
# 参数配置
# ------------------------
FRAME_RATE = 30
N_JOINTS = 8
DEVICE = 'cuda'  # 或 'cpu'
MODEL_PATH = 'best_gait_transformer.pth'

# ------------------------
# 模拟数据加载（这里你可以换成视频+深度读取）
# ------------------------
N_FRAMES = 50
H, W = 64, 64
rgb_frames = np.random.rand(N_FRAMES, H, W, 3)    # N x H x W x 3
depth_frames = np.random.rand(N_FRAMES, H, W)     # N x H x W

# ------------------------
# 1. 初始化3D骨架估计器
# ------------------------
pose2d = PoseEstimator()
pose3d = PoseEstimator3D(pose2d, depth_scale=1.0)
pose3d.set_camera_intrinsics(fx=500, fy=500, cx=W//2, cy=H//2)

skeleton_seq = pose3d.detect_batch_3d(rgb_frames, depth_frames)  # seq_len x n_joints x 3
print("3D Skeleton shape:", skeleton_seq.shape)

# ------------------------
# 2. 加载训练好的Transformer模型进行预测
# ------------------------
model = GaitTransformer(n_joints=N_JOINTS, in_dim=3, d_model=128, nhead=4, num_layers=3, out_dim=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

skeleton_tensor = torch.tensor(skeleton_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # 1 x T x J x 3
with torch.no_grad():
    predicted_params = model(skeleton_tensor).cpu().numpy()[0]  # T x out_dim
print("Predicted params shape:", predicted_params.shape)

# ------------------------
# 3. 步态周期分析
# ------------------------
analyzer = GaitAnalyzer(foot_z_indices=(4,5), frame_rate=FRAME_RATE)
gait_cycles = analyzer.analyze(skeleton_seq)
print("Detected gait cycles:", len(gait_cycles))
for i, c in enumerate(gait_cycles):
    print(f"{i+1}: {c['foot']} foot, frames {c['start_frame']}–{c['end_frame']}, "
          f"duration {c['duration']:.2f}s, step_length {c['step_length']:.3f}m")

# ------------------------
# 4. 可视化
# ------------------------
# 4.1 足部Z轴曲线
foot_z = skeleton_seq[:, [4,5], 2]
plt.figure(figsize=(10,4))
plt.plot(foot_z[:,0], label='Left foot Z')
plt.plot(foot_z[:,1], label='Right foot Z')

# 标记foot contact
left_contact = foot_z[:,0] < 0.05
right_contact = foot_z[:,1] < 0.05
plt.scatter(np.where(left_contact)[0], foot_z[left_contact,0], color='red', marker='x')
plt.scatter(np.where(right_contact)[0], foot_z[right_contact,1], color='blue', marker='x')
plt.xlabel('Frame')
plt.ylabel('Foot Z (m)')
plt.title('Foot Z coordinate over time')
plt.legend()
plt.savefig('foot_z_curve.png', dpi=200)
plt.close()

# 4.2 步长柱状图
step_lengths = [c['step_length'] for c in gait_cycles]
plt.figure(figsize=(8,4))
plt.bar(range(len(step_lengths)), step_lengths)
plt.xlabel('Gait cycle index')
plt.ylabel('Step length (m)')
plt.title('Step length per gait cycle')
plt.savefig('step_length_bar.png', dpi=200)
plt.close()

# 4.3 3D骨架动画
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(0,2)

edges = [
    (0,2),(2,4),(4,6),  # 左腿
    (1,3),(3,5),(5,7),  # 右腿
    (0,1),(2,3),(4,5),(6,7)  # 横向连接
]

scat = ax.scatter([],[],[], c='g', s=30)
lines = [ax.plot([],[],[], c='r')[0] for _ in edges]

def init():
    scat._offsets3d = ([],[],[])
    for line in lines:
        line.set_data([],[])
        line.set_3d_properties([])
    return [scat]+lines

def update(frame):
    skel = skeleton_seq[frame]
    x, y, z = skel[:,0], skel[:,1], skel[:,2]
    scat._offsets3d = (x,y,z)
    for i, (a,b) in enumerate(edges):
        lines[i].set_data([x[a], x[b]], [y[a], y[b]])
        lines[i].set_3d_properties([z[a], z[b]])
    return [scat]+lines

ani = FuncAnimation(fig, update, frames=len(skeleton_seq), init_func=init, blit=False, interval=1000//FRAME_RATE)
writer = FFMpegWriter(fps=FRAME_RATE)
ani.save('skeleton3d_animation.mp4', writer=writer)
plt.close()
print("Visualization saved: foot_z_curve.png, step_length_bar.png, skeleton3d_animation.mp4")
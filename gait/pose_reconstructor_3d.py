import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CameraParam:
    def __init__(self):
        self.fx = 410.7592468261719
        self.fy = 410.7592468261719
        self.cx = 423.3375244140625
        self.cy = 240.66250610351562

        self.R = np.eye(3)
        self.t = np.zeros((3,1))


class PoseReconstructor3D:

    def __init__(self, cam_param, pose2d, save_dir="output_3d"):
        self.cam = cam_param
        self.pose2d = pose2d

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.frame_id = 0
        self.all_3d = []
        self.prev_depths = None  # 上一帧每个关节深度

    # =========================
    # 深度滤波
    # =========================
    def get_depth(self, depth, u, v):

        h, w = depth.shape

        if u < 0 or v < 0 or u >= w or v >= h:
            return 0

        patch = depth[max(0,v-1):v+2, max(0,u-1):u+2]
        patch = patch[patch > 0]

        if len(patch) == 0:
            return 0

        return np.median(patch)

    # =========================
    # 2D → 3D
    # =========================
    def uv_to_xyz(self, u, v, d):

        Z = d / 1000.0
        if Z <= 0:
            return np.array([0,0,0])

        X = (u - self.cam.cx) * Z / self.cam.fx

        # ⭐ 核心修正：Y取负（图像坐标 → 世界坐标）
        Y = -(v - self.cam.cy) * Z / self.cam.fy

        return np.array([X, Y, Z])

    def cam_to_world(self, xyz):
        xyz = xyz.reshape(3,1)
        return (self.cam.R @ xyz + self.cam.t).flatten()

    # =========================
    # 重建
    # =========================
    def reconstruct(self, rgb, depth):

        skeleton_2d, skeleton_world = self.pose2d.detect(rgb)
        if skeleton_2d is None:
            return None

        skeleton_3d = []
        current_depths = np.zeros(len(skeleton_2d))  # ⭐ 记录本帧深度

        # -----------------------------
        # 判断左右髋哪侧在前
        # -----------------------------
        left_hip_c = skeleton_2d[1,2]
        right_hip_c = skeleton_2d[2,2]

        if left_hip_c >= right_hip_c:
            front_side = "left"
            back_side = "right"
        else:
            front_side = "right"
            back_side = "left"

        # 左右关节索引
        left_idxs = [1,3,5]   # hip, knee, ankle
        right_idxs = [2,4,6]

        # -----------------------------
        # 重建每个关节
        # -----------------------------
        for i, (u,v,c) in enumerate(skeleton_2d):
            u, v = int(u), int(v)

            # ===== 1️⃣ 当前深度 =====
            d = self.get_depth(depth, u, v)

            use_temporal = False

            # -----------------------------
            # 判断是否需要修正（遮挡）
            # -----------------------------
            if back_side == "left" and i in left_idxs:
                need_fix = (c < 0.5)
                same_side = left_idxs
            elif back_side == "right" and i in right_idxs:
                need_fix = (c < 0.5)
                same_side = right_idxs
            else:
                need_fix = False

            # -----------------------------
            # ⭐ 遮挡处理
            # -----------------------------
            if need_fix:

                # ===== 2️⃣ 空间补偿 =====
                d_vals = []
                for idx in same_side:
                    uu, vv = int(skeleton_2d[idx,0]), int(skeleton_2d[idx,1])
                    dd = self.get_depth(depth, uu, vv)
                    if dd > 0:
                        d_vals.append(dd)

                if len(d_vals) > 0:
                    d = np.median(d_vals)

                else:
                    # ===== 3️⃣ 时间补偿 ⭐核心新增 =====
                    if self.prev_depths is not None:
                        prev_d = self.prev_depths[i]
                        if prev_d > 0:
                            d = prev_d
                            use_temporal = True

            # -----------------------------
            # ⭐ 时间平滑（进一步稳定）
            # -----------------------------
            if self.prev_depths is not None:
                prev_d = self.prev_depths[i]
                if prev_d > 0 and d > 0:
                    alpha = 0.7 if use_temporal else 0.5
                    d = alpha * prev_d + (1 - alpha) * d

            current_depths[i] = d

            # -----------------------------
            # 转3D
            # -----------------------------
            xyz = self.uv_to_xyz(u, v, d)
            xyz = self.cam_to_world(xyz)

            skeleton_3d.append(xyz)

        # 更新历史深度
        self.prev_depths = current_depths.copy()

        skeleton_3d = np.array(skeleton_3d)
        self.all_3d.append(skeleton_3d)

        return skeleton_3d

        
    # =========================
    # 保存3D图片
    # =========================
    def save_3d_plot(self, skeleton_3d):

        fig = plt.figure(figsize=(8,8))  # ⭐ 放大画布
        ax = fig.add_subplot(111, projection='3d')

        # =========================
        # ⭐ 1. 中位数中心（关键！）
        # =========================
        center = np.median(skeleton_3d, axis=0)

        # 平移到中心
        centered = skeleton_3d - center

        # =========================
        # ⭐ 2. 轴映射（保持你原来的）
        # =========================
        xs = centered[:,0]*0.7
        ys = centered[:,2]   
        zs = centered[:,1]   

        # =========================
        # 关键点颜色
        # =========================
        left_idxs = {1,3,5,7,9}
        right_idxs = {2,4,6,8,10}

        for i in range(len(xs)):
            if i in left_idxs:
                color = 'r'
            elif i in right_idxs:
                color = 'b'
            else:
                color = 'g'

            ax.scatter(xs[i], ys[i], zs[i], c=color)

        # =========================
        # 骨架连接
        # =========================
        edges = [
            (0,1),(0,2),
            (1,3),(3,5),(5,7),(7,9),
            (2,4),(4,6),(6,8),(8,10)
        ]

        for i,j in edges:

            if i in left_idxs and j in left_idxs:
                color = 'r'
            elif i in right_idxs and j in right_idxs:
                color = 'b'
            else:
                color = 'g'

            ax.plot(
                [xs[i],xs[j]],
                [ys[i],ys[j]],
                [zs[i],zs[j]],
                c=color,
                linewidth=2
            )

        # =========================
        # ⭐ 3. 固定显示范围（不再动态变）
        # =========================
        fixed_range = 1.0   # ⭐ 你可以调这个（单位：米）

        ax.set_xlim(-fixed_range, fixed_range)
        ax.set_ylim(-fixed_range, fixed_range)
        ax.set_zlim(-fixed_range, fixed_range)

        # =========================
        # ⭐ 4. 固定比例（关键！）
        # =========================
        #ax.set_box_aspect([1,1,1])  # 保证xyz比例一致

        # =========================
        # 视角
        # =========================
        ax.view_init(elev=20, azim=-60)

        ax.set_xlabel("X")
        ax.set_ylabel("Z (forward)")
        ax.set_zlabel("Y (up)")

        save_path = os.path.join(self.save_dir, f"{self.frame_id:06d}.png")
        plt.savefig(save_path)
        plt.close(fig)

        self.frame_id += 1

    # =========================
    # 保存3D数据
    # =========================
    def save_all(self, path="skeleton_3d.npy"):
        np.save(path, np.array(self.all_3d, dtype=object))

    # =========================
    # ⭐ 新增：生成视频
    # =========================
    def make_video(self, timestamp_path, output_path="3d.mp4"):

        # ===== 读取时间戳 =====
        timestamps = []
        with open(timestamp_path) as f:
            for line in f:
                _, t = line.split()
                timestamps.append(float(t))

        # ===== 计算真实FPS =====
        fps_list = []
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                fps_list.append(1.0 / dt)

        fps = sum(fps_list) / len(fps_list)
        print(f"Real FPS: {fps:.2f}")

        # ===== 读取图片 =====
        files = sorted(os.listdir(self.save_dir))
        files = [f for f in files if f.endswith(".png")]

        if len(files) == 0:
            print("No images.")
            return

        first_img = cv2.imread(os.path.join(self.save_dir, files[0]))
        h, w = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print("Generating video (timestamp-based FPS)...")

        for f in files:
            img = cv2.imread(os.path.join(self.save_dir, f))
            writer.write(img)

        writer.release()

        print(f"Saved video: {output_path}")
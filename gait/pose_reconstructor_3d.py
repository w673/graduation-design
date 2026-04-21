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

        self.prev_depths = None
        self.prev_3d = None
        self.prev_confidences = None
        self.fixed_scale = None


        # ⭐ 骨架拓扑
        self.edges = [
            (0,1),(0,2),
            (1,3),(3,5),(5,7),(7,9),
            (2,4),(4,6),(6,8),(8,10)
        ]

    def _mean_uv(self, skeleton, indices):
        points = skeleton[indices, :2]
        return np.mean(points, axis=0)

    def _normalize_vec(self, vec, fallback=None):
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)

        if norm > 1e-6:
            return vec / norm

        if fallback is None:
            return None

        fallback = np.asarray(fallback, dtype=float)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm > 1e-6:
            return fallback / fallback_norm

        return None

    def _sample_xyz_from_depth(self, depth, u, v, clustered=False):
        u_i = int(round(u))
        v_i = int(round(v))

        if clustered:
            d = self.get_depth_clustered(depth, u_i, v_i)
        else:
            d = self.get_depth(depth, u_i, v_i)

        xyz = self.uv_to_xyz(u, v, d)
        return self.cam_to_world(xyz)

    def _build_pose_basis(self, points):
        pelvis = points[0].astype(float)
        left_hip = points[1].astype(float)
        right_hip = points[2].astype(float)
        left_knee = points[3].astype(float)
        right_knee = points[4].astype(float)
        left_ankle = points[5].astype(float)
        right_ankle = points[6].astype(float)
        left_heel = points[7].astype(float)
        right_heel = points[8].astype(float)
        left_toe = points[9].astype(float)
        right_toe = points[10].astype(float)

        lateral = self._normalize_vec(right_hip - left_hip)
        if lateral is None:
            lateral = self._normalize_vec(right_knee - left_knee, fallback=np.array([1.0, 0.0, 0.0]))

        foot_center = np.mean(points[[5, 6, 7, 8, 9, 10]], axis=0)
        vertical = self._normalize_vec(pelvis - foot_center, fallback=np.array([0.0, 1.0, 0.0]))

        forward = np.cross(vertical, lateral)
        forward = self._normalize_vec(forward, fallback=np.array([0.0, 0.0, 1.0]))

        toe_dir = self._normalize_vec((left_toe - left_heel) + (right_toe - right_heel))
        if toe_dir is None:
            toe_dir = self._normalize_vec(left_toe - left_heel, fallback=np.array([0.0, 0.0, 1.0]))
        if np.dot(forward, toe_dir) < 0:
            forward = -forward

        lateral = self._normalize_vec(np.cross(vertical, forward), fallback=lateral)
        vertical = self._normalize_vec(np.cross(forward, lateral), fallback=vertical)

        basis = np.stack([lateral, vertical, forward], axis=1)
        return pelvis, basis

    def _project_to_basis(self, points, origin, basis):
        return (points - origin) @ basis

    def _lift_from_basis(self, points_local, origin, basis):
        return points_local @ basis.T + origin

    def _reconstruct_from_pit_topology(self, skeleton_pit, depth):
        pit_points = skeleton_pit[:, :3].astype(float).copy()
        depth_h, depth_w = depth.shape[:2]

        def to_px(point_2d):
            px = point_2d.astype(float).copy()
            px[0] *= depth_w
            px[1] *= depth_h
            return px

        pelvis_px = to_px(skeleton_pit[0, :2])
        left_hip_px = to_px(skeleton_pit[1, :2])
        right_hip_px = to_px(skeleton_pit[2, :2])
        left_knee_px = to_px(skeleton_pit[3, :2])
        right_knee_px = to_px(skeleton_pit[4, :2])
        left_ankle_px = to_px(skeleton_pit[5, :2])
        right_ankle_px = to_px(skeleton_pit[6, :2])
        left_heel_px = to_px(skeleton_pit[7, :2])
        right_heel_px = to_px(skeleton_pit[8, :2])
        left_toe_px = to_px(skeleton_pit[9, :2])
        right_toe_px = to_px(skeleton_pit[10, :2])

        pelvis_anchor = self._sample_xyz_from_depth(depth, pelvis_px[0], pelvis_px[1], clustered=False)
        left_hip_anchor = self._sample_xyz_from_depth(depth, left_hip_px[0], left_hip_px[1], clustered=False)
        right_hip_anchor = self._sample_xyz_from_depth(depth, right_hip_px[0], right_hip_px[1], clustered=False)
        left_knee_anchor = self._sample_xyz_from_depth(depth, left_knee_px[0], left_knee_px[1], clustered=False)
        right_knee_anchor = self._sample_xyz_from_depth(depth, right_knee_px[0], right_knee_px[1], clustered=False)
        left_ankle_anchor = self._sample_xyz_from_depth(depth, left_ankle_px[0], left_ankle_px[1], clustered=True)
        right_ankle_anchor = self._sample_xyz_from_depth(depth, right_ankle_px[0], right_ankle_px[1], clustered=True)
        left_heel_anchor = self._sample_xyz_from_depth(depth, left_heel_px[0], left_heel_px[1], clustered=True)
        right_heel_anchor = self._sample_xyz_from_depth(depth, right_heel_px[0], right_heel_px[1], clustered=True)
        left_toe_anchor = self._sample_xyz_from_depth(depth, left_toe_px[0], left_toe_px[1], clustered=True)
        right_toe_anchor = self._sample_xyz_from_depth(depth, right_toe_px[0], right_toe_px[1], clustered=True)

        anchors = [
            pelvis_anchor,
            left_hip_anchor,
            right_hip_anchor,
            left_knee_anchor,
            right_knee_anchor,
            left_ankle_anchor,
            right_ankle_anchor,
            left_heel_anchor,
            right_heel_anchor,
            left_toe_anchor,
            right_toe_anchor,
        ]

        if any(np.all(anchor == 0) for anchor in anchors):
            return pit_points

        source_origin, source_basis = self._build_pose_basis(pit_points)

        source_local = self._project_to_basis(pit_points, source_origin, source_basis)
        target_points = np.vstack([
            pelvis_anchor,
            left_hip_anchor,
            right_hip_anchor,
            left_knee_anchor,
            right_knee_anchor,
            left_ankle_anchor,
            right_ankle_anchor,
            left_heel_anchor,
            right_heel_anchor,
            left_toe_anchor,
            right_toe_anchor,
        ])
        target_local = self._project_to_basis(target_points, pelvis_anchor, source_basis)

        axis_scales = [1.0, 1.0, 1.0]

        lateral_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        lateral_ratios = []
        for left_idx, right_idx in lateral_pairs:
            src_delta = source_local[right_idx, 0] - source_local[left_idx, 0]
            tgt_delta = target_local[right_idx, 0] - target_local[left_idx, 0]
            if abs(src_delta) > 1e-6 and abs(tgt_delta) > 1e-6:
                lateral_ratios.append(abs(tgt_delta) / abs(src_delta))
        if len(lateral_ratios) > 0:
            axis_scales[0] = float(np.clip(np.median(lateral_ratios), 0.5, 2.5))

        source_foot_center = np.mean(source_local[[5, 6, 7, 8, 9, 10]], axis=0)
        target_foot_center = np.mean(target_local[[5, 6, 7, 8, 9, 10]], axis=0)
        vertical_src = source_foot_center[1] - source_local[0, 1]
        vertical_tgt = target_foot_center[1] - target_local[0, 1]
        if abs(vertical_src) > 1e-6 and abs(vertical_tgt) > 1e-6:
            axis_scales[1] = float(np.clip(abs(vertical_tgt) / abs(vertical_src), 0.5, 2.5))

        forward_pairs = [(7, 9), (8, 10)]
        forward_ratios = []
        for heel_idx, toe_idx in forward_pairs:
            src_delta = source_local[toe_idx, 2] - source_local[heel_idx, 2]
            tgt_delta = target_local[toe_idx, 2] - target_local[heel_idx, 2]
            if abs(src_delta) > 1e-6 and abs(tgt_delta) > 1e-6:
                forward_ratios.append(abs(tgt_delta) / abs(src_delta))
        if len(forward_ratios) > 0:
            axis_scales[2] = float(np.clip(np.median(forward_ratios), 0.5, 2.5))

        scale = float(np.clip((axis_scales[0] * axis_scales[1] * axis_scales[2]) ** (1.0 / 3.0), 0.5, 2.5))
        if self.fixed_scale is None:
            self.fixed_scale = scale
        scale = self.fixed_scale

        scaled_local = source_local * scale

        reconstructed = self._lift_from_basis(scaled_local, pelvis_anchor, source_basis)
        reconstructed[0, :3] = pelvis_anchor

        return reconstructed

    # =========================
    # 普通深度（原始）
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
    # ⭐ 新增：脚部深度聚类
    # =========================
    def get_depth_clustered(self, depth, u, v, window=5):

        h, w = depth.shape

        if u < 0 or v < 0 or u >= w or v >= h:
            return 0

        patch = depth[
            max(0, v-window):min(h, v+window+1),
            max(0, u-window):min(w, u+window+1)
        ]

        vals = patch.flatten()
        vals = vals[vals > 0]

        if len(vals) < 5:
            return 0

        vals = np.sort(vals)

        # ⭐ 滑窗找最密集区域
        cluster_size = max(5, len(vals)//5)
        best_cluster = []
        min_range = float("inf")

        for i in range(len(vals) - cluster_size):
            window_vals = vals[i:i+cluster_size]
            r = window_vals[-1] - window_vals[0]

            if r < min_range:
                min_range = r
                best_cluster = window_vals

        if len(best_cluster) == 0:
            return 0

        return np.median(best_cluster)

    # =========================
    # 2D → 3D
    # =========================
    def uv_to_xyz(self, u, v, d):
        Z = d / 1000.0
        if Z <= 0:
            return np.array([0,0,0])

        X = (u - self.cam.cx) * Z / self.cam.fx
        Y = -(v - self.cam.cy) * Z / self.cam.fy

        return np.array([X, Y, Z])

    def cam_to_world(self, xyz):
        xyz = xyz.reshape(3,1)
        return (self.cam.R @ xyz + self.cam.t).flatten()

    # =========================
    # 重建
    # =========================
    def reconstruct(self, rgb, depth):

        skeleton_2d, skeleton_pit = self.pose2d.detect(rgb)
        if skeleton_2d is None:
            return None, None

        skeleton_3d = self._reconstruct_from_pit_topology(skeleton_pit, depth)

        self.prev_3d = skeleton_3d.copy()

        self.all_3d.append(skeleton_3d)

        return skeleton_3d, skeleton_pit

    # =========================
    # 保存3D图片（保持你原逻辑）
    # =========================
    def save_3d_plot(self, skeleton_3d):

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        joint_names = [
            "pelvis",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_toe",
            "right_toe",
        ]

        center = np.median(skeleton_3d, axis=0)
        centered = skeleton_3d - center

        xs = centered[:,0]
        ys = centered[:,2]
        zs = -centered[:,1]

        left_idxs = {1,3,5,7,9}
        right_idxs = {2,4,6,8,10}

        for i in range(len(xs)):
            if i in left_idxs:
                color = 'r'
            elif i in right_idxs:
                color = 'b'
            else:
                color = 'g'

            ax.scatter(xs[i], ys[i], zs[i], c=color, s=28)
            ax.text(
                xs[i], ys[i], zs[i],
                f"{i}:{joint_names[i]}",
                fontsize=8,
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=1.0)
            )

        for i,j in self.edges:
            if i in left_idxs and j in left_idxs:
                color = 'r'
            elif i in right_idxs and j in right_idxs:
                color = 'b'
            else:
                color = 'g'

            ax.plot([xs[i],xs[j]],
                    [ys[i],ys[j]],
                    [zs[i],zs[j]],
                    c=color, linewidth=2)

        fixed_range = 1.0
        ax.set_xlim(-fixed_range, fixed_range)
        ax.set_ylim(-fixed_range, fixed_range)
        ax.set_zlim(-fixed_range, fixed_range)

        ax.view_init(elev=0, azim=-80)

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

        save_path = os.path.join(self.save_dir, f"{self.frame_id:06d}.png")
        plt.savefig(save_path)
        plt.close(fig)
        self.frame_id += 1

    def save_all(self, path="skeleton_3d.npy"):
        np.save(path, np.array(self.all_3d, dtype=object))

    def make_video(self, timestamp_path, output_path="3d.mp4"):

        timestamps = []
        with open(timestamp_path) as f:
            for line in f:
                _, t = line.split()
                timestamps.append(float(t))

        fps = np.mean([1/(timestamps[i]-timestamps[i-1]) 
                       for i in range(1,len(timestamps)) 
                       if timestamps[i]>timestamps[i-1]])

        files = sorted([f for f in os.listdir(self.save_dir) if f.endswith(".png")])

        first = cv2.imread(os.path.join(self.save_dir, files[0]))
        h, w = first.shape[:2]

        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (w,h))

        for f in files:
            img = cv2.imread(os.path.join(self.save_dir, f))
            writer.write(img)

        writer.release()
        print(f"Saved video: {output_path}")
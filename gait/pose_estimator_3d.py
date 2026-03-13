import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器无窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

from pose_estimator import PoseEstimator


class PoseEstimator3D:
    """
    在已有2D PoseEstimator基础上，结合深度图生成3D骨架，并支持保存3D动画
    skeleton顺序:
    0 pelvis
    1 left_hip
    2 right_hip
    3 left_knee
    4 right_knee
    5 left_ankle
    6 right_ankle
    7 left_toe
    8 right_toe
    """

    def __init__(self, pose_estimator: PoseEstimator, depth_scale=1.0):

        self.pose2d = pose_estimator
        self.depth_scale = depth_scale

        # camera intrinsics
        self.camera_intrinsics = None

    # ---------------------------------------------------
    # 设置相机内参
    # ---------------------------------------------------
    def set_camera_intrinsics(self, fx, fy, cx, cy):

        self.camera_intrinsics = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }

    # ---------------------------------------------------
    # pixel → 3D
    # ---------------------------------------------------
    def _pixel_to_3d(self, u, v, depth):

        if self.camera_intrinsics is None:
            return np.array([u, v, depth])

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    # ---------------------------------------------------
    # 单帧3D检测
    # ---------------------------------------------------
    def detect_3d(self, image, depth_map):

        # 2D skeleton (9x3)
        skeleton2d = self.pose2d.detect(image)

        skeleton3d = []

        for kpt in skeleton2d:

            u, v = int(kpt[0]), int(kpt[1])

            h, w = depth_map.shape

            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)

            depth = depth_map[v, u] * self.depth_scale

            xyz = self._pixel_to_3d(u, v, depth)

            skeleton3d.append(xyz)

        return np.array(skeleton3d)

    # ---------------------------------------------------
    # 批量检测
    # ---------------------------------------------------
    def detect_batch_3d(self, images, depth_maps):

        skeletons = []

        for img, depth in zip(images, depth_maps):

            skel3d = self.detect_3d(img, depth)

            skeletons.append(skel3d)

        return np.array(skeletons)

    # ---------------------------------------------------
    # 保存3D骨架动画
    # ---------------------------------------------------
    def save_3d_animation(self, skeletons, out_file='skeleton3d.mp4', interval=50):

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(0, 2)

        # skeleton edges (9 joints)
        edges = [

            # pelvis → hips
            (0, 1),
            (0, 2),

            # left leg
            (1, 3),
            (3, 5),
            (5, 7),

            # right leg
            (2, 4),
            (4, 6),
            (6, 8)
        ]

        scat = ax.scatter([], [], [], c='g', s=30)

        lines = [ax.plot([], [], [], c='r')[0] for _ in edges]

        # --------------------------------
        # init
        # --------------------------------
        def init():

            scat._offsets3d = ([], [], [])

            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])

            return [scat] + lines

        # --------------------------------
        # update
        # --------------------------------
        def update(frame):

            skel = skeletons[frame]

            x = skel[:, 0]
            y = skel[:, 1]
            z = skel[:, 2]

            scat._offsets3d = (x, y, z)

            for i, (a, b) in enumerate(edges):

                lines[i].set_data([x[a], x[b]], [y[a], y[b]])
                lines[i].set_3d_properties([z[a], z[b]])

            return [scat] + lines

        ani = FuncAnimation(
            fig,
            update,
            frames=len(skeletons),
            init_func=init,
            blit=False,
            interval=interval
        )

        writer = FFMpegWriter(fps=1000 // interval)

        ani.save(out_file, writer=writer)

        plt.close(fig)
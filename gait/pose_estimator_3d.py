import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器无窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

from pose_estimator import PoseEstimator  # 复用你的2D PoseEstimator

class PoseEstimator3D:
    """
    在已有2D PoseEstimator基础上，结合深度图生成3D骨架，并支持保存3D动画
    """
    def __init__(self, pose_estimator: PoseEstimator, depth_scale=1.0):
        self.pose2d = pose_estimator
        self.depth_scale = depth_scale
        self.camera_intrinsics = None  # fx, fy, cx, cy

    def set_camera_intrinsics(self, fx, fy, cx, cy):
        self.camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

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

    def detect_3d(self, image, depth_map):
        skeleton2d = self.pose2d.detect(image)  # 8x3
        skeleton3d = []
        for kpt in skeleton2d:
            u, v = int(kpt[0]), int(kpt[1])
            h, w = depth_map.shape
            u = np.clip(u, 0, w-1)
            v = np.clip(v, 0, h-1)
            depth = depth_map[v, u] * self.depth_scale
            xyz = self._pixel_to_3d(u, v, depth)
            skeleton3d.append(xyz)
        return np.array(skeleton3d)

    def detect_batch_3d(self, images, depth_maps):
        skeletons = []
        for img, depth in zip(images, depth_maps):
            skel3d = self.detect_3d(img, depth)
            skeletons.append(skel3d)
        return np.array(skeletons)

    def save_3d_animation(self, skeletons, out_file='skeleton3d.mp4', interval=50):
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
            skel = skeletons[frame]
            x, y, z = skel[:,0], skel[:,1], skel[:,2]
            scat._offsets3d = (x,y,z)
            for i, (a,b) in enumerate(edges):
                lines[i].set_data([x[a], x[b]], [y[a], y[b]])
                lines[i].set_3d_properties([z[a], z[b]])
            return [scat]+lines

        ani = FuncAnimation(fig, update, frames=len(skeletons),
                            init_func=init, blit=False, interval=interval)
        writer = FFMpegWriter(fps=1000//interval)
        ani.save(out_file, writer=writer)
        plt.close(fig)
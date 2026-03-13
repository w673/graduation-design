import numpy as np


class GaitPreprocessor:

    def normalize_view(self, skeletons):

        skeletons = skeletons.copy()

        T = skeletons.shape[0]

        # ------------------------------------------------
        # 只用第一帧计算旋转
        # ------------------------------------------------

        sk = skeletons[0].copy()

        pelvis = sk[0]

        sk = sk - pelvis

        left_hip = sk[1]
        right_hip = sk[2]

        # ------------------------------------------------
        # Step1: hip axis -> X
        # ------------------------------------------------

        hip_vec = right_hip - left_hip
        hip_vec = hip_vec / np.linalg.norm(hip_vec)

        target_x = np.array([1, 0, 0])

        R1 = self.rotation_matrix(hip_vec, target_x)

        sk = sk @ R1.T

        # ------------------------------------------------
        # Step2: forward direction -> Z
        # ------------------------------------------------

        left_leg = sk[5] - sk[1]
        right_leg = sk[6] - sk[2]

        forward = (left_leg + right_leg) / 2
        forward = forward / np.linalg.norm(forward)

        target_z = np.array([0, 0, 1])

        R2 = self.rotation_matrix(forward, target_z)

        R = R2 @ R1

        # ------------------------------------------------
        # 应用到所有帧
        # ------------------------------------------------

        for t in range(T):

            sk = skeletons[t]

            pelvis = sk[0]

            sk = sk - pelvis

            sk = sk @ R.T

            skeletons[t] = sk

        return skeletons

    def rotation_matrix(self, a, b):

        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)

        if s < 1e-6:
            return np.eye(3)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        return R

    def skeleton_to_tokens(self, skeletons):

        T, J, C = skeletons.shape

        tokens = skeletons.reshape(T, J * C)

        return tokens
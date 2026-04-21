import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# 关键点定义
# -----------------------------
JOINT_NAMES = [
    "pelvis",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_toe","right_toe"
]

MP_JOINTS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_toe": 31,
    "right_toe": 32
}


class PoseEstimator2D:

    def __init__(self):

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='/data1/fwl_files/design/model/mediapipe/pose_landmarker_heavy.task'),
                running_mode=python.vision.RunningMode.VIDEO
            )
        )

        self.prev_skeleton = None  # 用于时间平滑
        self.timestamp = 0

    # -----------------------------
    # 主检测函数
    # -----------------------------
    def detect(self, image):

        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.pose.detect_for_video(mp_image, self.timestamp)
        self.timestamp += 33  # assume 30fps

        if not results.pose_landmarks:
            return None, None  # 返回两个 None

        landmarks = results.pose_landmarks[0]

        def get_xyzc(idx):
            lm = landmarks[idx]

            u = int(lm.x * w)
            v = int(lm.y * h)
            z = lm.z
            c = lm.visibility

            u = np.clip(u, 0, w-1)
            v = np.clip(v, 0, h-1)

            return np.array([u, v, z, c])

        def get_xyzc_pit(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z, lm.visibility])  # 单位米

        # -----------------------------
        # joints
        # -----------------------------
        hip_left = get_xyzc(MP_JOINTS["left_hip"])
        hip_right = get_xyzc(MP_JOINTS["right_hip"])
        knee_left = get_xyzc(MP_JOINTS["left_knee"])
        knee_right = get_xyzc(MP_JOINTS["right_knee"])
        ankle_left = get_xyzc(MP_JOINTS["left_ankle"])
        ankle_right = get_xyzc(MP_JOINTS["right_ankle"])
        heel_left = get_xyzc(MP_JOINTS["left_heel"])
        heel_right = get_xyzc(MP_JOINTS["right_heel"])
        toe_left = get_xyzc(MP_JOINTS["left_toe"])
        toe_right = get_xyzc(MP_JOINTS["right_toe"])

        # -----------------------------
        # pelvis
        # -----------------------------
        pelvis_xyz = (hip_left[:3] + hip_right[:3]) / 2
        pelvis_conf = min(hip_left[3], hip_right[3])
        pelvis = np.array([pelvis_xyz[0], pelvis_xyz[1], pelvis_xyz[2], pelvis_conf])

        pelvis_pit = (get_xyzc_pit(MP_JOINTS["left_hip"]) + 
                        get_xyzc_pit(MP_JOINTS["right_hip"])) / 2

        # -----------------------------
        # skeleton拼接
        # -----------------------------
        skeleton = np.vstack([
            pelvis,
            hip_left,
            hip_right,
            knee_left,
            knee_right,
            ankle_left,
            ankle_right,
            heel_left,
            heel_right,
            toe_left,
            toe_right
        ])

        skeleton_pit = np.vstack([
            pelvis_pit,
            get_xyzc_pit(MP_JOINTS["left_hip"]),
            get_xyzc_pit(MP_JOINTS["right_hip"]),
            get_xyzc_pit(MP_JOINTS["left_knee"]),
            get_xyzc_pit(MP_JOINTS["right_knee"]),
            get_xyzc_pit(MP_JOINTS["left_ankle"]),
            get_xyzc_pit(MP_JOINTS["right_ankle"]),
            get_xyzc_pit(MP_JOINTS["left_heel"]),
            get_xyzc_pit(MP_JOINTS["right_heel"]),
            get_xyzc_pit(MP_JOINTS["left_toe"]),
            get_xyzc_pit(MP_JOINTS["right_toe"])
        ])

        # -----------------------------
        # 时间平滑（可选但推荐）
        # -----------------------------
        if self.prev_skeleton is not None:
            skeleton_2d = 0.0 * self.prev_skeleton + 1.0 * skeleton

        self.prev_skeleton = skeleton

        # 返回两个数组：原2D skeleton + 世界坐标 skeleton
        return skeleton, skeleton_pit

    # -----------------------------
    # batch处理
    # -----------------------------
    def detect_batch(self, frames):

        results = []

        for frame in frames:
            skeleton = self.detect(frame)
            results.append(skeleton)

        return np.array(results)
    
    def draw_skeleton(self, image, skeleton, conf=0.3):

        img = image.copy()

        left_idxs = {1,3,5,7,9}
        right_idxs = {2,4,6,8,10}

        skeleton_edges = [
            (0,1),(0,2),
            (1,3),(3,5),(5,7),(7,9),
            (2,4),(4,6),(6,8),(8,10)
        ]

        # draw points
        for x,y,c in skeleton:
            if c > conf:
                cv2.circle(img, (int(x),int(y)), 5, (0,255,0), -1)

        # draw edges
        for i,j in skeleton_edges:
            if skeleton[i,2] > conf and skeleton[j,2] > conf:

                x1,y1 = skeleton[i][:2]
                x2,y2 = skeleton[j][:2]

                if i in left_idxs and j in left_idxs:
                    color = (0,0,255)
                elif i in right_idxs and j in right_idxs:
                    color = (255,0,0)
                else:
                    color = (0,255,0)

                cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2),), color, 2)

        return img
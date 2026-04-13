import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# 关键点定义（17点，兼容VideoPose3D）
# -----------------------------
JOINT_NAMES = [
    "pelvis",

    # 上半身
    "nose",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",

    # 下半身
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_toe","right_toe"
]

MP_JOINTS = {
    "nose": 0,

    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

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
                base_options=python.BaseOptions(
                    model_asset_path='/data1/fwl_files/design/model/mediapipe/pose_landmarker_heavy.task'
                ),
                running_mode=python.vision.RunningMode.VIDEO
            )
        )

        self.prev_skeleton = None
        self.timestamp = 0

    # -----------------------------
    # 主检测函数
    # -----------------------------
    def detect(self, image):

        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.pose.detect_for_video(mp_image, self.timestamp)
        self.timestamp += 33

        if not results.pose_landmarks:
            return None, None

        landmarks = results.pose_landmarks[0]
        world_landmarks = results.pose_world_landmarks[0]

        def get_xyc(idx):
            lm = landmarks[idx]
            u = int(lm.x * w)
            v = int(lm.y * h)
            c = lm.visibility

            u = np.clip(u, 0, w-1)
            v = np.clip(v, 0, h-1)

            return np.array([u, v, c])

        def get_xyz_world(idx):
            lm = world_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        # -----------------------------
        # 上半身
        # -----------------------------
        nose = get_xyc(MP_JOINTS["nose"])

        shoulder_left = get_xyc(MP_JOINTS["left_shoulder"])
        shoulder_right = get_xyc(MP_JOINTS["right_shoulder"])

        elbow_left = get_xyc(MP_JOINTS["left_elbow"])
        elbow_right = get_xyc(MP_JOINTS["right_elbow"])

        wrist_left = get_xyc(MP_JOINTS["left_wrist"])
        wrist_right = get_xyc(MP_JOINTS["right_wrist"])

        # -----------------------------
        # 下半身
        # -----------------------------
        hip_left = get_xyc(MP_JOINTS["left_hip"])
        hip_right = get_xyc(MP_JOINTS["right_hip"])

        knee_left = get_xyc(MP_JOINTS["left_knee"])
        knee_right = get_xyc(MP_JOINTS["right_knee"])

        ankle_left = get_xyc(MP_JOINTS["left_ankle"])
        ankle_right = get_xyc(MP_JOINTS["right_ankle"])

        heel_left = get_xyc(MP_JOINTS["left_heel"])
        heel_right = get_xyc(MP_JOINTS["right_heel"])

        toe_left = get_xyc(MP_JOINTS["left_toe"])
        toe_right = get_xyc(MP_JOINTS["right_toe"])

        # -----------------------------
        # pelvis
        # -----------------------------
        pelvis_xy = (hip_left[:2] + hip_right[:2]) / 2
        pelvis_conf = min(hip_left[2], hip_right[2])
        pelvis = np.array([pelvis_xy[0], pelvis_xy[1], pelvis_conf])

        pelvis_world = (get_xyz_world(MP_JOINTS["left_hip"]) +
                        get_xyz_world(MP_JOINTS["right_hip"])) / 2

        # -----------------------------
        # skeleton（17点）
        # -----------------------------
        skeleton = np.vstack([
            pelvis,

            # 上半身
            nose,
            shoulder_left, shoulder_right,
            elbow_left, elbow_right,
            wrist_left, wrist_right,

            # 下半身
            hip_left, hip_right,
            knee_left, knee_right,
            ankle_left, ankle_right,
            heel_left, heel_right,
            toe_left, toe_right
        ])

        skeleton_world = np.vstack([
            pelvis_world,

            get_xyz_world(MP_JOINTS["nose"]),
            get_xyz_world(MP_JOINTS["left_shoulder"]),
            get_xyz_world(MP_JOINTS["right_shoulder"]),
            get_xyz_world(MP_JOINTS["left_elbow"]),
            get_xyz_world(MP_JOINTS["right_elbow"]),
            get_xyz_world(MP_JOINTS["left_wrist"]),
            get_xyz_world(MP_JOINTS["right_wrist"]),

            get_xyz_world(MP_JOINTS["left_hip"]),
            get_xyz_world(MP_JOINTS["right_hip"]),
            get_xyz_world(MP_JOINTS["left_knee"]),
            get_xyz_world(MP_JOINTS["right_knee"]),
            get_xyz_world(MP_JOINTS["left_ankle"]),
            get_xyz_world(MP_JOINTS["right_ankle"]),
            get_xyz_world(MP_JOINTS["left_heel"]),
            get_xyz_world(MP_JOINTS["right_heel"]),
            get_xyz_world(MP_JOINTS["left_toe"]),
            get_xyz_world(MP_JOINTS["right_toe"])
        ])

        # -----------------------------
        # 时间平滑
        # -----------------------------
        if self.prev_skeleton is not None:
            alpha = 0.7
            skeleton = alpha * self.prev_skeleton + (1 - alpha) * skeleton

        self.prev_skeleton = skeleton

        return skeleton, skeleton_world

    # -----------------------------
    # batch处理
    # -----------------------------
    def detect_batch(self, frames):

        results = []

        for frame in frames:
            skeleton, _ = self.detect(frame)
            results.append(skeleton)

        return np.array(results)

    # -----------------------------
    # 可视化
    # -----------------------------
    def draw_skeleton(self, image, skeleton, conf=0.3):

        img = image.copy()

        skeleton_edges = [
            # 上半身
            (1,2),(2,4),(4,6),
            (1,3),(3,5),(5,7),

            # 躯干
            (0,8),(0,9),

            # 左腿
            (8,10),(10,12),(12,14),(14,16),

            # 右腿
            (9,11),(11,13),(13,15),(15,17)
        ]

        for x,y,c in skeleton:
            if c > conf:
                cv2.circle(img, (int(x),int(y)), 4, (0,255,0), -1)

        for i,j in skeleton_edges:
            if skeleton[i,2] > conf and skeleton[j,2] > conf:
                x1,y1 = skeleton[i][:2]
                x2,y2 = skeleton[j][:2]
                cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

        return img
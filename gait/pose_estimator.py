import numpy as np
import cv2
import torch


from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

# PackDetInputs 在 mmdet 中定义，我们要确保它在两个作用域里都被注册。
# 先切换到 mmdet 作用域注册一次，然后再切换回 mmpose 作用域。
from mmengine.registry import init_default_scope
init_default_scope('mmdet')
from mmdet.datasets.transforms.formatting import PackDetInputs as _PackDetInputs

# 切回 mmpose 以便后续 mmpose 导入使用
init_default_scope('mmpose')
from mmdet.datasets.transforms.formatting import PackDetInputs

from segment_anything import sam_model_registry, SamPredictor
# qualify imports so scripts running from workspace root can resolve submodules
from gait.toe_estimator import process_frame_footpoint

# body8 关键点顺序（RTMPose body8）：
# 0: nose, 1: left_shoulder, 2: right_shoulder, 3: left_elbow, 4: right_elbow,
# 5: left_hip, 6: right_hip, 7: left_knee, 8: right_knee, 9: left_ankle, 10: right_ankle
# 我们只使用下半身：hip, knee, ankle
BODY_JOINTS = [11, 12, 13, 14, 15, 16]  # left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

JOINT_NAMES = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_toe",
    "right_toe"
]


class PoseEstimator:

    def __init__(
        self,
        pose_config,
        pose_checkpoint,
        mmdet_config,
        mmdet_checkpoint,
        sam_checkpoint,
        device="cuda:0",
        mask_update_interval=10
    ):

        # -----------------------------
        # mmdet detection model
        # -----------------------------
        self.det_model = init_detector(
            mmdet_config,
            mmdet_checkpoint,
            device=device
        )

        # -----------------------------
        # pose model
        # -----------------------------
        self.pose_model = init_model(
            pose_config,
            pose_checkpoint,
            device=device
        )

        # -----------------------------
        # SAM segmentation
        # -----------------------------
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

        self.mask_update_interval = mask_update_interval
        self.current_mask = None
        self.frame_idx = 0

        # -----------------------------
        # toe tracking state
        # -----------------------------
        self.previous_frame_data = {
            'toe_left': None,
            'toe_right': None,
            'knee_left': None,
            'knee_right': None,
            'ank_left': None,
            'ank_right': None,
            'last_known_relative_angle_left': 0.0,
            'last_known_relative_angle_right': 0.0,
            'angular_velocity_left': 0.0,
            'angular_velocity_right': 0.0,
            'toe_knee_radius_left': 1.0,
            'toe_knee_radius_right': 1.0
        }


    # -----------------------------
    # SAM segmentation
    # -----------------------------
    def segment_person(self, image):
        self.predictor.set_image(image)
        h, w = image.shape[:2]
        input_point = np.array([[w/2, h/2]])
        input_label = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8)


    # -----------------------------
    # detect people with mmdet
    # -----------------------------
    def detect_person_bbox(self, image, score_thr=0.5):
        # make sure registry is in mmdet scope before building the pipeline
        from mmengine.registry import init_default_scope
        init_default_scope('mmdet')

        # mmdet 的返回类型在不同版本稍有变化
        result = inference_detector(self.det_model, image)
        # 旧版：list/tuple，每个元素是 ndarray of shape (N,5) [x1,y1,x2,y2,score]
        # 新版：DetDataSample，包含 pred_instances.bboxes (N,4) 和 scores (N,)
        if isinstance(result, (list, tuple)):
            bboxes = result[0]  # assume person is class 0
            if bboxes.shape[1] == 5:
                # old format: [x1,y1,x2,y2,score]
                person_bboxes = bboxes[bboxes[:, 4] > score_thr][:, :4]
            else:
                # unexpected, but assume no scores
                person_bboxes = bboxes
        else:
            # new format: DetDataSample
            if hasattr(result, 'pred_instances'):
                pts = result.pred_instances
                bboxes = pts.bboxes
                scores = pts.scores
                # convert to numpy if tensor
                try:
                    bboxes = bboxes.cpu().numpy()
                    scores = scores.cpu().numpy()
                except Exception:
                    bboxes = np.array(bboxes)
                    scores = np.array(scores)
                # filter by score
                valid = scores > score_thr
                person_bboxes = bboxes[valid]
            else:
                raise RuntimeError(f'unexpected detector output: {type(result)}')

        return person_bboxes


    # -----------------------------
    # pose detection
    # -----------------------------
    def detect(self, image):
        # 每 N 帧更新 mask
        if self.frame_idx % self.mask_update_interval == 0:
            self.current_mask = self.segment_person(image)
        processed_mask = self.current_mask
        self.frame_idx += 1
        if processed_mask is None:
            processed_mask = np.ones(image.shape[:2], dtype=np.uint8)

        # -----------------------------
        # detect person bbox
        # -----------------------------
        bboxes = self.detect_person_bbox(image)
        if len(bboxes) == 0:
            # fallback: full image
            bboxes = np.array([[0,0,image.shape[1], image.shape[0]]], dtype=np.float32)

        # 这里只取第一个检测到的人
        bbox = bboxes[0].reshape(1, 4)

        # -----------------------------
        # pose estimation
        # -----------------------------
        result = inference_topdown(
            self.pose_model,
            image,
            bbox
        )

        data_samples = merge_data_samples(result)
        keypoints = data_samples.pred_instances.keypoints[0]
        scores = data_samples.pred_instances.keypoint_scores[0]
        keypoints = np.concatenate([keypoints, scores[:,None]], axis=1)

        # 只保留body8前6个关节
        body_kpts = keypoints[BODY_JOINTS]

        # -----------------------------
        # leg joints
        # -----------------------------
        knee_left = body_kpts[2][:2]
        knee_right = body_kpts[3][:2]
        ank_left = body_kpts[4][:2]
        ank_right = body_kpts[5][:2]
        calf_length_left = np.linalg.norm(np.array(ank_left) - np.array(knee_left))
        calf_length_right = np.linalg.norm(np.array(ank_right) - np.array(knee_right))

        # -----------------------------
        # toe estimation
        # -----------------------------
        toe_left, toe_right, self.previous_frame_data = process_frame_footpoint(
            processed_mask,
            ank_left, knee_left, calf_length_left,
            ank_right, knee_right, calf_length_right,
            self.previous_frame_data
        )

        toe_left = np.array([toe_left[0], toe_left[1], 1.0])
        toe_right = np.array([toe_right[0], toe_right[1], 1.0])

        skeleton = np.vstack([body_kpts, toe_left, toe_right])
        return skeleton


    # -----------------------------
    # batch detection
    # -----------------------------
    def detect_batch(self, frames):
        results = []
        for frame in frames:
            skeleton = self.detect(frame)
            results.append(skeleton)
        return np.array(results)


# -----------------------------
# skeleton visualization
# -----------------------------
def draw_skeleton(image, skeleton, conf=0.3):
    img = image.copy()
    # 左右身体分别用不同颜色绘制：左侧红，右侧蓝
    left_idxs = {0,2,4,6}   # left_hip, left_knee, left_ankle, left_toe
    right_idxs = {1,3,5,7}  # right_hip, right_knee, right_ankle, right_toe

    # 连接关系可以根据应用适当调整，保留之前的结构
    skeleton_edges = [
        (0,1),(1,3),(3,5),
        (5,7),(0,2),(2,4),
        (4,6)
    ]

    # draw points
    for x,y,c in skeleton:
        if c > conf:
            cv2.circle(img, (int(x),int(y)), 5, (0,255,0), -1)

    # draw edges with color by side
    for i,j in skeleton_edges:
        if skeleton[i,2] > conf and skeleton[j,2] > conf:
            x1,y1 = skeleton[i][:2]
            x2,y2 = skeleton[j][:2]
            # determine color
            if i in left_idxs and j in left_idxs:
                color = (0,0,255)   # red for left body
            elif i in right_idxs and j in right_idxs:
                color = (255,0,0)   # blue for right body
            else:
                color = (0,255,0)   # cross or mixed -> green
            cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
    return img
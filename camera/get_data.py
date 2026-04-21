from pyorbbecsdk import *
import cv2
import numpy as np
import time
import os

# ===== 保存目录 =====
os.makedirs("data1/rgb_frames", exist_ok=True)
os.makedirs("data1/depth_frames", exist_ok=True)

timestamp_file = open("data1/timestamps.txt", "w")


def pick_video_profile(profile_list, candidates):
    for width, height, fmt, fps in candidates:
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            continue
    return profile_list.get_default_video_stream_profile()

# ===== 初始化 =====
pipeline = Pipeline()
config = Config()

color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

# 优先选择更高 FPS、且适合做对齐采集的 profile；找不到则回退默认 profile
color_profile = pick_video_profile(
    color_profiles,
    [
        (640, 480, OBFormat.RGB, 30),
        (1280, 720, OBFormat.RGB, 30),
        (1280, 720, OBFormat.MJPG, 30),
    ],
)
depth_profile = pick_video_profile(
    depth_profiles,
    [
        (640, 480, OBFormat.Y16, 30),
        (848, 480, OBFormat.Y16, 30),
    ],
)

config.enable_stream(color_profile)
config.enable_stream(depth_profile)
config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

try:
    pipeline.enable_frame_sync()
except Exception:
    pass

pipeline.start(config)

print(
    "Using color profile:",
    color_profile.get_width(),
    "x",
    color_profile.get_height(),
    "fps=",
    color_profile.get_fps(),
    "format=",
    color_profile.get_format(),
)
print(
    "Using depth profile:",
    depth_profile.get_width(),
    "x",
    depth_profile.get_height(),
    "fps=",
    depth_profile.get_fps(),
    "format=",
    depth_profile.get_format(),
)

# ===== SDK降噪 =====
device = pipeline.get_device()
device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL, True)
device.set_int_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_DIFF_INT, 512)
device.set_int_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_SPECKLE_SIZE_INT, 960)

# ===== 滤波链 =====
depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)
filter_list = depth_sensor.get_recommended_filters()

# ===== 对齐 =====
align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

print("Recording... Press ESC to stop")

frame_id = 0

while True:
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        continue

    frames = align_filter.process(frames)
    if not frames:
        continue

    frames = frames.as_frame_set()

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if color_frame is None or depth_frame is None:
        continue

    # ===== 时间戳（核心）=====
    timestamp = time.time()

    # ===== COLOR =====
    fmt = str(color_frame.get_format())
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if "MJPG" in fmt:
        color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        w = color_frame.get_width()
        h = color_frame.get_height()
        try:
            color = data.reshape((h, w, 3))
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        except:
            continue

    # ===== DEPTH =====
    depth_frame_filtered = depth_frame

    for post_filter in filter_list:
        if post_filter.is_enabled():
            depth_frame_filtered = post_filter.process(depth_frame_filtered)

    if hasattr(depth_frame_filtered, "as_video_frame"):
        depth_frame_filtered = depth_frame_filtered.as_video_frame()

    w_d = depth_frame_filtered.get_width()
    h_d = depth_frame_filtered.get_height()

    depth = np.frombuffer(depth_frame_filtered.get_data(), dtype=np.uint16)
    depth = depth.reshape((h_d, w_d))

    # 可视化
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)

    # ===== 保存 =====
    cv2.imwrite(f"data1/rgb_frames/{frame_id:06d}.png", color)
    cv2.imwrite(f"data1/depth_frames/{frame_id:06d}.png", depth_vis)
    np.save(f"data1/depth_frames/{frame_id:06d}.npy", depth)

    timestamp_file.write(f"{frame_id} {timestamp}\n")

    frame_id += 1

    cv2.imshow("color", color)
    cv2.imshow("depth_vis", depth_vis)

    if cv2.waitKey(1) == 27:
        break

# ===== 结束 =====
pipeline.stop()
cv2.destroyAllWindows()
timestamp_file.close()

print("Recording finished.")
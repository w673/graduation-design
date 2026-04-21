import cv2
import numpy as np
import shutil
from pathlib import Path

# -----------------------------
# 输入输出路径
# -----------------------------
rgb_folder = Path("D:/Demo/rgb_frames")
depth_folder = Path("D:/Demo/depth_frames")
timestamps_file = Path("D:/Demo/timestamps.txt")
skeleton_file = Path("D:/Demo/skeleton_3d.npy")
output_folder = Path("D:/Demo/labeled_dataset")
output_folder.mkdir(parents=True, exist_ok=True)

if not rgb_folder.exists():
    raise FileNotFoundError(f"RGB folder not found: {rgb_folder}")

rgb_files = sorted([f for f in rgb_folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
if not rgb_files:
    raise FileNotFoundError(f"No image files found in: {rgb_folder}")

num_total_frames = len(rgb_files)

# skeleton_files = sorted([f for f in skeleton_folder.iterdir() if f.suffix=='.npy'])

# # -----------------------------
# 第一步：选取有效帧区间
# -----------------------------
start_frame, end_frame = None, None
current_frame = 0

def mouse_select(event, x, y, flags, param):
    global start_frame, end_frame, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_frame is None:
            start_frame = current_frame  # 点击第一次为开始帧
        else:
            end_frame = current_frame  # 点击第二次为结束帧

cv2.namedWindow("Select valid range")
cv2.setMouseCallback("Select valid range", mouse_select)

while True:
    frame = cv2.imread(str(rgb_files[current_frame]))
    display = frame.copy()
    cv2.putText(display, f"Frame {current_frame}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(display, "Click to select start/end frame, q to finish", (10,70), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow("Select valid range", display)
    
    key = cv2.waitKey(50)
    if key == ord('d') and current_frame < len(rgb_files)-1:
        current_frame += 1
    elif key == ord('a') and current_frame > 0:
        current_frame -= 1
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

if start_frame is None or end_frame is None:
    print("No range selected, using all frames")
    start_frame, end_frame = 0, len(rgb_files)-1

if start_frame > end_frame:
    start_frame, end_frame = end_frame, start_frame

print(f"Selected frames: {start_frame} ~ {end_frame}")

# 只保留有效帧
rgb_files = rgb_files[start_frame:end_frame+1]
# skeleton_files = skeleton_files[start_frame:end_frame+1]

# -----------------------------
# 第二步：逐帧标注
# -----------------------------
button_labels = ["LH", "LTO", "RH", "RTO"]
none_label_id = len(button_labels)
num_buttons = len(button_labels)
button_height = 50
button_width = 100
button_margin = 10

buttons = []
for i in range(num_buttons):
    x1 = button_margin + i*(button_width+button_margin)
    y1 = 0
    x2 = x1 + button_width
    y2 = button_height
    buttons.append((x1, y1, x2, y2))

# -1 表示当前帧尚未标注，保存前会统一映射到 None
labels = np.full(len(rgb_files), -1, dtype=int)
current_frame = 0
clicked_label = None

def mouse_label(event, x, y, flags, param):
    global clicked_label
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, (x1, y1, x2, y2) in enumerate(buttons):
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_label = idx
                break

cv2.namedWindow("Label Frame")
cv2.setMouseCallback("Label Frame", mouse_label)

while current_frame < len(rgb_files):
    frame = cv2.imread(str(rgb_files[current_frame]))
    display = frame.copy()
    global_frame = start_frame + current_frame

    # 绘制按钮
    for idx, (x1, y1, x2, y2) in enumerate(buttons):
        color = (0,255,0) if labels[current_frame]==idx else (255,255,255)
        cv2.rectangle(display, (x1,y1), (x2,y2), color, -1)
        cv2.putText(display, button_labels[idx], (x1+10, y1+35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    frame_status = "Unlabeled" if labels[current_frame] == -1 else button_labels[labels[current_frame]]
    
    cv2.putText(display, f"Frame {current_frame}/{len(rgb_files)-1} (global: {global_frame})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.putText(display, f"Current label: {frame_status}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
    cv2.putText(display, "Click LH/LTO/RH/RTO to label; press d or s to skip; a back; q quit", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

    cv2.imshow("Label Frame", display)
    key = cv2.waitKey(50)

    if clicked_label is not None:
        labels[current_frame] = clicked_label
        clicked_label = None
        current_frame += 1

    if key == ord('a') and current_frame > 0:
        current_frame -= 1
    elif key == ord('d') or key == ord('s'):
        current_frame += 1
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# -----------------------------
# 保存标注与 skeleton 有效片段
# -----------------------------
# 未标注帧统一设为 None 类别
labels[labels == -1] = none_label_id
np.save(output_folder / "labels_valid.npy", labels)

if skeleton_file.exists():
    skeleton = np.load(skeleton_file, allow_pickle=True)
    if isinstance(skeleton, np.ndarray) and skeleton.dtype == object:
        try:
            skeleton = skeleton.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to convert object skeleton array to numeric type: {e}")

    if skeleton.ndim != 3 or skeleton.shape[1:] != (11, 3):
        raise ValueError(
            f"Skeleton shape must be (N, 11, 3), but got {skeleton.shape}."
        )

    if skeleton.shape[0] <= end_frame:
        raise ValueError(
            f"Skeleton frame count ({skeleton.shape[0]}) is not enough for selected end frame ({end_frame})."
        )

    valid_skeleton = skeleton[start_frame:end_frame+1]
    if valid_skeleton.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Valid skeleton length ({valid_skeleton.shape[0]}) does not match label length ({labels.shape[0]})."
        )

    np.save(output_folder / "skeleton_valid.npy", valid_skeleton)
else:
    print(f"Skeleton file not found, skip saving valid skeleton: {skeleton_file}")

# -----------------------------
# 保存有效区间对应的RGB/Depth/时间戳
# -----------------------------
valid_rgb_folder = output_folder / "rgb_frames"
if valid_rgb_folder.exists():
    shutil.rmtree(valid_rgb_folder)
valid_rgb_folder.mkdir(parents=True, exist_ok=True)

for i, src_path in enumerate(rgb_files):
    new_name = f"{i:06d}{src_path.suffix.lower()}"
    shutil.copy2(src_path, valid_rgb_folder / new_name)

if depth_folder.exists():
    depth_files = sorted([f for f in depth_folder.iterdir() if f.is_file()])
    depth_ext_priority = [".npy", ".png", ".jpg", ".jpeg"]
    depth_files_by_ext = {
        ext: sorted([f for f in depth_files if f.suffix.lower() == ext], key=lambda p: p.name)
        for ext in depth_ext_priority
    }

    selected_depth_ext = None
    for ext in depth_ext_priority:
        if len(depth_files_by_ext[ext]) >= num_total_frames:
            selected_depth_ext = ext
            break

    if selected_depth_ext is None:
        ext_count_msg = ", ".join([f"{ext}:{len(depth_files_by_ext[ext])}" for ext in depth_ext_priority])
        raise ValueError(
            f"No depth extension has enough frames for alignment with RGB ({num_total_frames}). Counts -> {ext_count_msg}"
        )

    print(f"Using depth files with extension: {selected_depth_ext}")

    valid_rgb_stems = [p.stem for p in rgb_files]
    depth_map = {p.stem: p for p in depth_files_by_ext[selected_depth_ext]}
    missing_stems = [stem for stem in valid_rgb_stems if stem not in depth_map]
    if missing_stems:
        raise ValueError(
            f"Missing depth frames for selected RGB stems (first 10): {missing_stems[:10]}"
        )

    valid_depth_files = [depth_map[stem] for stem in valid_rgb_stems]
    valid_depth_folder = output_folder / "depth_frames"
    if valid_depth_folder.exists():
        shutil.rmtree(valid_depth_folder)
    valid_depth_folder.mkdir(parents=True, exist_ok=True)

    for i, src_path in enumerate(valid_depth_files):
        new_name = f"{i:06d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, valid_depth_folder / new_name)
else:
    print(f"Depth folder not found, skip saving valid depth frames: {depth_folder}")

if timestamps_file.exists():
    with timestamps_file.open("r", encoding="utf-8") as f:
        timestamp_lines = f.readlines()

    if len(timestamp_lines) < num_total_frames:
        raise ValueError(
            f"Timestamp count ({len(timestamp_lines)}) is less than RGB frame count ({num_total_frames})."
        )

    valid_timestamp_lines = timestamp_lines[start_frame:end_frame+1]
    renumbered_timestamp_lines = []
    for i, line in enumerate(valid_timestamp_lines):
        stripped = line.strip()
        if not stripped:
            renumbered_timestamp_lines.append(f"{i}\n")
            continue

        parts = stripped.split(maxsplit=1)
        if len(parts) == 1:
            renumbered_timestamp_lines.append(f"{i}\n")
        else:
            renumbered_timestamp_lines.append(f"{i} {parts[1]}\n")

    valid_timestamps_file = output_folder / "timestamps_valid.txt"
    with valid_timestamps_file.open("w", encoding="utf-8") as f:
        f.writelines(renumbered_timestamp_lines)
else:
    print(f"Timestamp file not found, skip saving valid timestamps: {timestamps_file}")

print("标注完成，保存到:", output_folder)
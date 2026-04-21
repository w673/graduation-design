from pyorbbecsdk import *

def get_depth_intrinsics():

    pipeline = Pipeline()
    config = Config()

    # 启动深度流
    config.enable_stream(OBStreamType.DEPTH_STREAM)
    pipeline.start(config)

    # ⭐ 关键修改：用 SENSOR
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    if profile_list is None:
        print("Failed to get depth profile list")
        return

    profile = profile_list.get_default_video_stream_profile()

    if profile is None:
        print("Failed to get default profile")
        return

    intrinsic = profile.get_intrinsic()

    print("===== Depth Intrinsics =====")
    print("fx:", intrinsic.fx)
    print("fy:", intrinsic.fy)
    print("cx:", intrinsic.cx)
    print("cy:", intrinsic.cy)

    print("Resolution:", profile.get_width(), "x", profile.get_height())
    print("FPS:", profile.get_fps())
    print("Format:", profile.get_format())

    pipeline.stop()


if __name__ == "__main__":
    get_depth_intrinsics()
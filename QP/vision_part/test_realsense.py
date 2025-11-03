import pyrealsense2 as rs

# 파이프라인 및 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트리밍 시작
pipeline.start(config)

# 프레임 수신
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# intrinsics (내부 파라미터) 추출
profile = color_frame.get_profile()
intr = profile.as_video_stream_profile().get_intrinsics()

print("Width:", intr.width)
print("Height:", intr.height)
print("Fx:", intr.fx)
print("Fy:", intr.fy)
print("Cx:", intr.ppx)
print("Cy:", intr.ppy)
print("Distortion model:", intr.model)
print("Distortion coefficients:", intr.coeffs)

pipeline.stop()

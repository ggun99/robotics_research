import cv2
import os

# AVI 파일 경로
video_path = "output.avi"

if not os.path.exists(video_path):
    print("Error: Video file does not exist.")
else:
    print("Video file found.")


# VideoCapture 객체 생성 (AVI 파일 열기)
cap = cv2.VideoCapture(video_path)

# 비디오 파일이 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 비디오 정보 출력 (프레임 너비, 높이, FPS 등)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)

print(f"Frame Width: {frame_width}")
print(f"Frame Height: {frame_height}")
print(f"FPS: {fps}")

# 비디오 재생
while True:
    ret, frame = cap.read()  # 비디오에서 프레임 읽기

    if not ret:
        print("End of video.")
        break

    # 프레임을 윈도우 창에 표시
    cv2.imshow('AVI Video', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

import pyzed.sl as sl
import cv2


cam = sl.Camera()
init = sl.InitParameters()
# init.image_size = sl.RESOLUTION.HD720  # 1280x720 해상도
init.depth_mode = sl.DEPTH_MODE.NONE # Set configuration parameters for the ZED

status = cam.open(init) 
if status != sl.ERROR_CODE.SUCCESS: 
    print("Camera Open", status, "Exit program.")
    exit(1)
    
# 카메라 정보 가져오기
camera_info = cam.get_camera_information()
# fps = cam.get_current_fps()
w = camera_info.camera_configuration.resolution.width
h = camera_info.camera_configuration.resolution.height
fps = cam.get_init_parameters().camera_fps
print(f'fps:{fps}, w:{w}, h:{h}')

runtime = sl.RuntimeParameters()
# 프레임 저장을 위한 OpenCV VideoWriter 설정
# 비디오 코덱 설정 (XVID, MJPG, H264 등 가능)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # XVID 코덱
out = cv2.VideoWriter('left_camera_output.mp4', fourcc, fps, (w, h))  # 비디오 저장 파일 및 속성 설정
# 이미지 객체 초기화
left_image = sl.Mat()

while True:
    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS : # Check that a new image is successfully acquired
        
        # 왼쪽 카메라 영상 추출
        cam.retrieve_image(left_image, sl.VIEW.LEFT)
        
        # PyZED에서 추출한 이미지를 NumPy 배열로 변환
        left_frame = left_image.get_data()
        left_frame_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGBA2RGB)
        print(f"==>> left_frame.shape: {left_frame_bgr.shape}")

        # OpenCV로 영상을 저장
        out.write(left_frame_bgr)

        # OpenCV로 영상을 화면에 표시
        cv2.imshow("Left Camera", left_frame_bgr)
        # cv2.imwrite("zzz.jpg",left_frame)

        # 'q' 키를 누르면 녹화 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        print("Error grabbing frame")
        break

out.release()
cam.close()
cv2.destroyAllWindows()

import cv2
import pyzed.sl as sl

# ZED 카메라 초기화
zed = sl.Camera()

# 초기화 매개변수 설정
init_params = sl.InitParameters()
init_params.set_from_svo_file("one.svo2")  # SVO 파일 경로 설정
init_params.coordinate_units = sl.UNIT.METER  # 거리 단위 설정
init_params.svo_real_time_mode = False  # 실시간이 아닌 SVO 파일 재생 모드 설정

# ZED 카메라 열기
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"ZED 카메라 열기에 실패했습니다. 오류: {err}")
    exit(1)

# 이미지와 뎁스 데이터를 가져오기 위한 설정
image = sl.Mat()

# SVO 파일에서 프레임을 읽어와 OpenCV에서 처리
runtime_params = sl.RuntimeParameters()
while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # SVO 파일에서 왼쪽 이미지 프레임 추출
    zed.retrieve_image(image, sl.VIEW.LEFT)  # VIEW.LEFT 또는 VIEW.RIGHT 사용 가능
    frame = image.get_data()  # OpenCV에서 사용할 수 있는 형식으로 변환

    # OpenCV로 프레임을 표시
    cv2.imshow("SVO Frame", frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ZED 카메라 및 리소스 해제
zed.close()
cv2.destroyAllWindows()

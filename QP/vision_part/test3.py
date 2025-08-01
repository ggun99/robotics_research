import numpy as np
import cv2
import mediapipe as mp
import pyzed.sl as sl

# ZED 초기화
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
zed.open(init_params)

# 이미지 가져오기 위한 변수
image_zed = sl.Mat()
runtime_params = sl.RuntimeParameters()

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        
        # numpy array로 변환
        frame = image_zed.get_data()
        
        # BGRA → RGB로 변환 (MediaPipe는 RGB 기대)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # MediaPipe로 손 인식
        results = hands.process(frame_rgb)

        # 시각화 (원하면 BGR로 다시 변환)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp.solutions.drawing_utils.draw_landmarks(
                #     frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                x = int(wrist.x * frame_rgb.shape[1])
                y = int(wrist.y * frame_rgb.shape[0])
                # hand_positions.append((x, y))
                cv2.circle(frame_bgr, (x, y), 6, (0, 255, 255), -1)
        cv2.imshow("ZED + MediaPipe", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()

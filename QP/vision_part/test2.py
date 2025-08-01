import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import pyzed.sl as sl

# Create a Camera object
zed = sl.Camera()
# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)    
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution to 640x480
init_params.camera_fps = 30  # Optional: Set FPS (e.g., 30 FPS)
# Open the camera   
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
    print("Camera Open : "+repr(status)+". Exit program.")
# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()
image = sl.Mat(zed.get_camera_information().camera_configuration.resolution.width, zed.get_camera_information().camera_configuration.resolution.height, sl.MAT_TYPE.U8_C4)
depth = sl.Mat()
point_cloud = sl.Mat()
mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75,4.0,0))


# Load YOLOv8
yolo_model = YOLO("yolov8n.pt")  # 가벼운 버전 사용 (n = nano)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        imgnp = image.get_data()

    if imgnp is None:
        print("Image not loaded. Please check the image path or URL.")
        
    # resize_img = cv.resize(imgnp, (640, 480))  # Resize for speed

    imgrgb = cv2.cvtColor(imgnp, cv2.COLOR_BGRA2RGB)
    frame_bgr = cv2.cvtColor(imgrgb, cv2.COLOR_RGB2BGR)

    # ret, frame = cap.read()
    # if not ret:
    #     break

    # Resize for speed
    # img = cv2.resize(frame, (2560, 1440))
    # img = imgrg
    img_rgb = imgrgb # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Image shape:", img_rgb.shape)
    
    # 1. Detect people and objects
    yolo_results = yolo_model(img_rgb, verbose=False)[0]
    person_boxes = []
    object_boxes = []
    for box in yolo_results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:  # person
            person_boxes.append((x1, y1, x2, y2))
        elif cls != 0:  # other object
            object_boxes.append(((x1+x2)//2, (y1+y2)//2))

    # 2. Hand detection
    results = hands.process(img_rgb)
    hand_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]  # wrist landmark
            print("Wrist landmark:", wrist)

            x = int(wrist.x * imgrgb.shape[1])
            y = int(wrist.y * imgrgb.shape[0])
            hand_positions.append((x, y))
            cv2.circle(frame_bgr, (x, y), 6, (0, 255, 255), -1)

    # # 3. Estimate which hand is holding an object
    # for hand in hand_positions:
    #     closest_obj = None
    #     min_dist = float('inf')
    #     for obj in object_boxes:
    #         dist = np.linalg.norm(np.array(hand) - np.array(obj))
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest_obj = obj

    #     if closest_obj and min_dist < 80:  # 거리 threshold
    #         cv2.line(img, hand, closest_obj, (0, 0, 255), 2)
    #         cv2.putText(img, "HOLDING", (hand[0]+10, hand[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4. Draw person boxes
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLO + Hand Tracking", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
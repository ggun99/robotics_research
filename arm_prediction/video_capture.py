########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import argparse 
import os 
import cv2

cam = sl.Camera()

#Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)

signal(SIGINT, handler)

def main():
    
    init = sl.InitParameters()
    # init.image_size = sl.RESOLUTION.HD720  # 1280x720 해상도
    init.depth_mode = sl.DEPTH_MODE.NONE # Set configuration parameters for the ZED

    status = cam.open(init) 
    if status != sl.ERROR_CODE.SUCCESS: 
        print("Camera Open", status, "Exit program.")
        exit(1)
        
    recording_param = sl.RecordingParameters(opt.output_svo_file, sl.SVO_COMPRESSION_MODE.H264) # Enable recording with the filename specified in argument
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)
    # 카메라 정보 가져오기
    camera_info = cam.get_camera_information()
    # fps = cam.get_current_fps()
    w = camera_info.camera_configuration.resolution.width
    h = camera_info.camera_configuration.resolution.height
    fps = cam.get_init_parameters().camera_fps
    print(f'fps:{fps}, w:{w}, h:{h}')
    
    runtime = sl.RuntimeParameters()
    print("SVO is Recording, use Ctrl-C to stop.") # Start recording SVO, stop with Ctrl-C command
    frames_recorded = 0
    # 프레임 저장을 위한 OpenCV VideoWriter 설정
    # 비디오 코덱 설정 (XVID, MJPG, H264 등 가능)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # XVID 코덱
    out = cv2.VideoWriter('left_camera_output.mp4', fourcc, fps, (w, h))  # 비디오 저장 파일 및 속성 설정
    # 이미지 객체 초기화
    left_image = sl.Mat()

    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS : # Check that a new image is successfully acquired
            frames_recorded += 1
            print("Frame count: " + str(frames_recorded), end="\r")
            # 왼쪽 카메라 영상 추출
            cam.retrieve_image(left_image, sl.VIEW.LEFT)
            
            # PyZED에서 추출한 이미지를 NumPy 배열로 변환
            left_frame = left_image.get_data()

            # OpenCV로 영상을 저장
            out.write(left_frame)

            # OpenCV로 영상을 화면에 표시
            cv2.imshow("Left Camera", left_frame)

            # 'q' 키를 누르면 녹화 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            print("Error grabbing frame")
            break
    cam.close()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_svo_file', type=str, help='Path to the SVO file that will be written', required= True)
    opt = parser.parse_args()
    if not opt.output_svo_file.endswith(".svo") and not opt.output_svo_file.endswith(".svo2"): 
        print("--output_svo_file parameter should be a .svo file but is not : ",opt.output_svo_file,"Exit program.")
        exit()
    main()
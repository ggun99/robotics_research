import numpy as np
import cv2

class ARUCOGenerate:

    def __init__(self) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.size = (5, 7)
        self.markerLength = 0.071
        self.markerSeparation = 0.01
        self.board = cv2.aruco.GridBoard(self.size, self.markerLength, self.markerSeparation, self.dictionary, None)
        image = self.board.generateImage(outSize=(400, 500), marginSize=10, borderBits=1)

        import matplotlib.pyplot as plt
        plt.imshow(image, cmap="gray")
        plt.show()

class ARUCOBoardPose:
    # https://docs.opencv.org/4.9.0/db/da9/tutorial_aruco_board_detection.html
    def __init__(self) -> None:
        # detection
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.size = (5, 7)
        self.markerLength = 0.071
        self.markerSeparation = 0.01
        self.board = cv2.aruco.GridBoard(self.size, self.markerLength, self.markerSeparation, self.dictionary, None)
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detectorParams)

    def run(self, camera_k, camera_d, imgraw):
        corners, ids, rej = self.detector.detectMarkers(imgraw)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(imgraw, corners, ids)  # aruco corner

            objPoints, imgPoints = self.board.matchImagePoints(corners, ids, None, None)

            retval, rvc, tvc = cv2.solvePnP(objPoints, imgPoints, camera_k, camera_d, None, None, False)
            R, _ = cv2.Rodrigues(rvc)

            if objPoints is not None:
                cv2.drawFrameAxes(imgraw, camera_k, camera_d, rvc, tvc, 0.1, 3)

            return tvc, R
        return None
    
class ARUCORobotPose:
    # https://docs.opencv.org/4.9.0/db/da9/tutorial_aruco_board_detection.html
    def __init__(self) -> None:
        # detection
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.size = (5, 7)
        self.markerLength = 0.06
        self.markerSeparation = 0.005
        self.board = cv2.aruco.GridBoard(self.size, self.markerLength, self.markerSeparation, self.dictionary, None)
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detectorParams)

    def run(self, camera_k, camera_d, imgraw):
        corners, ids, rej = self.detector.detectMarkers(imgraw)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(imgraw, corners, ids)  # aruco corner

            objPoints, imgPoints = self.board.matchImagePoints(corners, ids, None, None)

            # ✅ 점 개수 체크 추가
            if objPoints is not None and imgPoints is not None:
                num_points = len(objPoints)
                if num_points >= 4:  # solvePnP 최소 요구사항
                    retval, rvc, tvc = cv2.solvePnP(objPoints, imgPoints, camera_k, camera_d, None, None, False)
                    R, _ = cv2.Rodrigues(rvc)

                    cv2.drawFrameAxes(imgraw, camera_k, camera_d, rvc, tvc, 0.1, 3)
                    return tvc, R, len(ids)
                else:
                    # 점이 부족할 때 (디버깅용)
                    cv2.putText(imgraw, f"Not enough points: {num_points}/4", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    return None
            else:
                cv2.putText(imgraw, "No matching points found", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                return None
        else: 
            return None

if __name__ == "__main__":
    ARUCOGenerate()
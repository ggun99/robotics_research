import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import time
import rtde_control
import rtde_receive

class UR5eController:
    def __init__(self, robot_ip="192.160.0.4"):  # 실제 UR5e IP로 변경
        """UR5e RTDE 컨트롤러 초기화"""
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        time.sleep(1)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.is_connected = True
        print(f"✅ UR5e 연결 성공: {robot_ip}")
    
    def get_current_joint_positions(self):
        """현재 조인트 위치 반환"""
        if not self.is_connected:
            return None
        return self.rtde_r.getActualQ()
    
    def move_joint1_by_degrees(self, degrees):
        """첫 번째 조인트를 지정된 각도만큼 회전 (최대 단순화)"""
        if not self.is_connected:
            print("❌ 로봇이 연결되지 않았습니다.")
            return False
        
        try:
            # 현재 조인트 위치
            current_q = self.rtde_r.getActualQ()
            if current_q is None:
                return False
            
            # 목표 위치 계산
            target_q = current_q.copy()
            target_q[0] += np.radians(degrees)
            
            print(f"🤖 Joint1 회전: {degrees}° (현재: {np.degrees(current_q[0]):.1f}° → 목표: {np.degrees(target_q[0]):.1f}°)")
            
            # ✅ 움직임 명령 (매우 느리게)
            success = self.rtde_c.moveJ(target_q, 0.05, 0.02)  # 매우 느린 속도
            print(f"📋 moveJ 명령 결과: {success}")
            
            if success:
                # ✅ 고정 시간 대기 (충분히 길게)
                print("⏳ 10초 대기 중...")
                time.sleep(2.5)  # 10초 고정 대기
                
                print("✅ Joint1 회전 완료 (시간 기반)")
                return True
            else:
                print("❌ Joint1 회전 실패")
                return False
                
        except Exception as e:
            print(f"❌ 오류: {e}")
            return False
    
    def get_current_pose(self):
        """현재 TCP 포즈 반환"""
        if not self.is_connected:
            return None
        return self.rtde_r.getActualTCPPose()
    
    def is_robot_ready(self):
        """로봇이 움직임 준비 상태인지 확인"""
        if not self.is_connected:
            return False
        return self.rtde_r.getRobotStatus() == 0  # 0 = 정지 상태
    
    def disconnect(self):
        """연결 해제"""
        if self.is_connected:
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
            self.is_connected = False
            print("🔌 UR5e 연결 해제됨")

class UR5e_Robot_Controller(Node):
    def __init__(self):
        super().__init__('ur5e_robot_controller')
        
        # ✅ ROS2 구독자 - 데이터 수집 완료 신호 받기
        self.data_complete_sub = self.create_subscription(
            Bool, '/robot_auto/data_collected', self.data_collected_callback, 10)
        
        # ✅ ROS2 퍼블리셔 - 상태 알림용
        self.status_pub = self.create_publisher(String, '/robot_auto/status', 10)
        self.robot_ready_pub = self.create_publisher(Bool, '/robot_auto/robot_ready', 10)
        
        # ✅ UR5e 컨트롤러 초기화
        self.ur5e = UR5eController("192.160.0.4")  # 실제 IP로 변경
        
        # ✅ 자동 시퀀스 관련 변수
        self.auto_mode = False
        self.current_rotation = 0  # 현재 회전 각도
        self.target_rotations = 360 // 5  # 72개 위치 (5도씩 72번)
        self.rotation_step = 5  # 5도씩 회전
        self.sequence_count = 0
        self.waiting_for_data = False
        
        # 타이머 (상태 확인용)
        self.create_timer(1.0, self.status_timer)
        
        self.get_logger().info("🤖 UR5e 로봇 컨트롤러 초기화 완료")
        self.get_logger().info(f"   - 회전 스텝: {self.rotation_step}도")
        self.get_logger().info(f"   - 총 회전 수: {self.target_rotations}회")
        self.get_logger().info(f"   - UR5e 연결: {'✅' if self.ur5e.is_connected else '❌'}")
        self.get_logger().info("   - 'A' 키를 눌러 자동 모드 시작")
        
    def data_collected_callback(self, msg: Bool):
        """데이터 수집 완료 신호 받았을 때 콜백"""
        if msg.data and self.auto_mode and self.waiting_for_data:
            self.get_logger().info("📊 데이터 수집 완료 신호 받음 - 다음 위치로 이동")
            self.waiting_for_data = False
            self.move_to_next_position()
    
    def start_auto_sequence(self):
        """자동 시퀀스 시작"""
        if not self.ur5e.is_connected:
            self.get_logger().error("❌ UR5e가 연결되지 않았습니다!")
            return
        
        self.auto_mode = True
        self.current_rotation = 0
        self.sequence_count = 0
        self.waiting_for_data = False
        
        # 상태 알림
        status_msg = String()
        status_msg.data = "auto_started"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info("🚀 자동 데이터 수집 시퀀스 시작!")
        self.get_logger().info(f"   총 {self.target_rotations}개 위치에서 데이터 수집 예정")
        
        # 첫 번째 위치에서 데이터 수집 신호 전송
        self.signal_ready_for_data()
    
    def stop_auto_sequence(self):
        """자동 시퀀스 정지"""
        self.auto_mode = False
        self.waiting_for_data = False
        
        # 상태 알림
        status_msg = String()
        status_msg.data = "auto_stopped"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info("⏹️  자동 데이터 수집 시퀀스 정지됨")
    
    def move_to_next_position(self):
        """다음 위치로 이동"""
        if not self.auto_mode:
            return
        
        # 시퀀스 완료 확인
        if self.sequence_count >= self.target_rotations:
            self.get_logger().info("🎉 전체 시퀀스 완료!")
            self.stop_auto_sequence()
            return
        
        # 로봇 움직임
        success = self.ur5e.move_joint1_by_degrees(self.rotation_step)
        
        if success:
            self.current_rotation += self.rotation_step
            self.sequence_count += 1
            
            self.get_logger().info(f"🔄 위치 {self.sequence_count}/{self.target_rotations} "
                                 f"(총 회전: {self.current_rotation}도)")
            
            # 이동 완료 후 데이터 수집 준비 신호
            time.sleep(1.0)  # 로봇 안정화 대기
            self.signal_ready_for_data()
        else:
            self.get_logger().error("❌ 로봇 이동 실패 - 시퀀스 정지")
            self.stop_auto_sequence()
    
    def signal_ready_for_data(self):
        """데이터 수집 준비 완료 신호"""
        self.waiting_for_data = True
        
        # 로봇 준비 완료 신호 전송
        ready_msg = Bool()
        ready_msg.data = True
        self.robot_ready_pub.publish(ready_msg)
        
        # 상태 알림
        status_msg = String()
        status_msg.data = f"ready_for_data_{self.sequence_count}"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f"📍 위치 {self.sequence_count} - 데이터 수집 대기 중...")
    
    def status_timer(self):
        """주기적 상태 확인"""
        if self.ur5e.is_connected:
            # 현재 조인트 위치 확인
            current_q = self.ur5e.get_current_joint_positions()
            if current_q is not None:
                joint1_deg = np.degrees(current_q[0])
                
                # 상태 정보 (5초마다 출력)
                if hasattr(self, '_status_counter'):
                    self._status_counter += 1
                else:
                    self._status_counter = 1
                
                if self._status_counter % 5 == 0:  # 5초마다
                    self.get_logger().info(f"📊 상태: 자동모드={'✅' if self.auto_mode else '❌'}, "
                                         f"Joint1={joint1_deg:.1f}°, "
                                         f"진행={self.sequence_count}/{self.target_rotations}")
    
    def manual_move_joint1(self, degrees):
        """수동으로 Joint1 이동"""
        if self.auto_mode:
            self.get_logger().warn("⚠️  자동 모드 중에는 수동 이동이 불가능합니다.")
            return
        
        success = self.ur5e.move_joint1_by_degrees(degrees)
        if success:
            self.get_logger().info(f"✅ 수동 이동 완료: Joint1 {degrees}도 회전")
        else:
            self.get_logger().error(f"❌ 수동 이동 실패: Joint1 {degrees}도 회전")
    
    def print_help(self):
        """도움말 출력"""
        help_text = """
        🤖 UR5e 로봇 컨트롤러 명령어:
        
        A/a  - 자동 시퀀스 시작 (5도씩 72번 회전)
        S/s  - 자동 시퀀스 정지
        +    - Joint1 +5도 회전 (수동)
        -    - Joint1 -5도 회전 (수동)
        R/r  - 현재 상태 출력
        H/h  - 이 도움말 출력
        Q/q  - 프로그램 종료
        """
        print(help_text)
        self.get_logger().info("도움말이 출력되었습니다.")

def main(args=None):
    rclpy.init(args=args)
    robot_controller = UR5e_Robot_Controller()
    
    # 도움말 출력
    robot_controller.print_help()
    
    # 키보드 입력을 위한 스레드 시작
    import threading
    
    def keyboard_input():
        """키보드 입력 처리"""
        while rclpy.ok():
            try:
                key = input().strip().lower()
                
                if key == 'q':
                    print("프로그램 종료 요청...")
                    rclpy.shutdown()
                    break
                elif key == 'a':
                    robot_controller.start_auto_sequence()
                elif key == 's':
                    robot_controller.stop_auto_sequence()
                elif key == '+':
                    robot_controller.manual_move_joint1(5)
                elif key == '-':
                    robot_controller.manual_move_joint1(-5)
                elif key == 'r':
                    if robot_controller.ur5e.is_connected:
                        current_q = robot_controller.ur5e.get_current_joint_positions()
                        if current_q:
                            joint1_deg = np.degrees(current_q[0])
                            print(f"📊 현재 상태:")
                            print(f"   Joint1: {joint1_deg:.1f}도")
                            print(f"   자동모드: {'활성' if robot_controller.auto_mode else '비활성'}")
                            print(f"   진행상황: {robot_controller.sequence_count}/{robot_controller.target_rotations}")
                    else:
                        print("❌ 로봇이 연결되지 않았습니다.")
                elif key == 'h':
                    robot_controller.print_help()
                else:
                    print("❓ 알 수 없는 명령어입니다. 'h' 를 입력하여 도움말을 확인하세요.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"입력 처리 오류: {e}")
    
    # 키보드 입력 스레드 시작
    input_thread = threading.Thread(target=keyboard_input, daemon=True)
    input_thread.start()
    
    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")
    finally:
        # UR5e 연결 해제
        robot_controller.ur5e.disconnect()
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
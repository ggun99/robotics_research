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
    
    def move_joint2_by_degrees(self, degrees):
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
            target_q[1] += np.radians(degrees)
            
            print(f"🤖 Joint2 회전: {degrees}° (현재: {np.degrees(current_q[1]):.1f}° → 목표: {np.degrees(target_q[1]):.1f}°)")
            
            # ✅ 움직임 명령 (매우 느리게)
            success = self.rtde_c.moveJ(target_q, 0.5, 0.03)  # 매우 느린 속도
            print(f"📋 moveJ 명령 결과: {success}")
            time.sleep(3.5)
            current_q_ = self.rtde_r.getActualQ()
            success2 = True if np.abs(np.degrees(current_q_[1]) - np.degrees(target_q[1])) <= 0.1 else False
            print(f"📋 목표 도달 여부: {success2}")
            if success2:
                # ✅ 고정 시간 대기 (충분히 길게)
                print("⏳ 10초 대기 중...")
                # time.sleep(5)  # 10초 고정 대기
                
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
        
        # ✅ 자동 시퀀스 관련 변수 (한 바퀴마다 복귀, 총 cycles 반복)
        self.auto_mode = False
        self.rotation_step = 5                          # degrees per step
        self.positions_per_rotation = 90 // self.rotation_step
        self.cycles = 5                                 # 총 360° 사이클 수
        self.cycle_index = 0
        self.current_rotation = 0.0                     # 현재 사이클 내 누적 각도 (deg)
        self.sequence_count = 0                         # 포지션 인덱스 (0..positions_per_rotation-1)
        self.waiting_for_data = False
        self.start_joint1 = None                         # 시작 각도(라디안) 기록 -> 절대 복귀에 사용
        # 타이머 (상태 확인용)
        self.create_timer(1.0, self.status_timer)
        
        self.get_logger().info("🤖 UR5e 로봇 컨트롤러 초기화 완료")
        self.get_logger().info(f"   - 회전 스텝: {self.rotation_step}도")
        self.get_logger().info(f"   - 총 회전 수: {self.cycles}회")
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
        # 시작 조인트1 각도 기록 (절대 복귀에 사용)
        q = self.ur5e.get_current_joint_positions()
        self.start_joint1 = q[0] if q is not None else None
        self.current_rotation = 0.0
        self.sequence_count = 0
        self.cycle_index = 0
        self.waiting_for_data = False
        
        
        # 상태 알림
        status_msg = String()
        status_msg.data = "auto_started"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info("🚀 자동 데이터 수집 시퀀스 시작!")
        # self.get_logger().info(f"   총 {self.73}개 위치에서 데이터 수집 예정")
        
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
        
        # 현재 사이클에서 모든 포지션 수집했는지 확인
        if self.sequence_count >= self.positions_per_rotation:
            self.get_logger().info(f"🔁 사이클 {self.cycle_index+1}/{self.cycles} 완료 (360°)")
            # 복귀: 기록된 시작각으로 절대 복귀 (현재->start 차이를 degree로 보내서 상대 이동)
            if self.start_joint1 is not None:
                cur_q = self.ur5e.get_current_joint_positions()
                if cur_q is None:
                    self.get_logger().error("❌ 현재 관절 정보를 읽을 수 없음 - 시퀀스 중단")
                    self.stop_auto_sequence()
                    return
                delta_deg = np.degrees(self.start_joint1 - cur_q[0])
                self.get_logger().info(f"↩️  사이클 복귀: Joint1 {delta_deg:.1f}° 이동 (절대 복귀)")
                ok = self.ur5e.move_joint1_by_degrees(delta_deg)
                if not ok:
                    self.get_logger().error("❌ 복귀 실패 - 시퀀스 중단")
                    self.stop_auto_sequence()
                    return
            else:
                self.get_logger().warn("⚠️ 시작 각도 정보 없음 — 복귀 건너뜀")

            self.cycle_index += 1
            if self.cycle_index >= self.cycles:
                self.get_logger().info("🎉 전체 사이클 완료!")
                self.stop_auto_sequence()
                return

            # 다음 사이클 준비
            self.current_rotation = 0.0
            self.sequence_count = 0
            time.sleep(1.0)
            # 다음 사이클 첫 포지션 준비 신호
            self.signal_ready_for_data()
            return

        # 정상 스텝 이동 (다음 포지션으로)
        success = self.ur5e.move_joint1_by_degrees(self.rotation_step)
        if success:
            self.current_rotation += self.rotation_step
            # sequence_count는 0..positions_per_rotation-1 로 유지: publish할 때 그대로 사용
            self.sequence_count += 1
            self.get_logger().info(f"🔄 위치 {self.sequence_count}/{self.positions_per_rotation} "
                                   f"(사이클 {self.cycle_index+1}/{self.cycles}, 총 회전: {self.current_rotation}°)")
            time.sleep(1.0)
            self.signal_ready_for_data()
        else:
            self.get_logger().error("❌ 로봇 이동 실패 - 시퀀스 정지")
            self.stop_auto_sequence()
    
    def move_to_j1_0_position(self):
        degrees = 0  # Joint1을 0도로 이동시키기 위한 각도 계산
        success = self.ur5e.move_joint1_by_degrees(degrees)
        current_q_ = self.rtde_r.getActualQ()
        target_q = degrees
        # 목표 도달 폴링
        timeout_sec = 30.0
        poll_interval = 0.5
        tolerance_deg = 0.1
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            current_q_ = self.rtde_r.getActualQ()
            if current_q_ is None:
                time.sleep(poll_interval)
                continue
            diff_deg = np.abs(np.degrees(current_q_[0]) - np.degrees(target_q[0]))
            if diff_deg <= tolerance_deg:
                print(f"📋 목표 도달 확인 (오차 {diff_deg:.3f}°) ")
                success2 = True
            time.sleep(poll_interval)
        # success2 = True if np.abs(np.degrees(current_q_[0]) - (target_q)) <= 0.1 else False
        if success2:
            self.get_logger().info(f"✅ 수동 이동 완료: Joint1 {degrees}도 회전")
        else:
            self.get_logger().error(f"❌ 수동 이동 실패: Joint1 {degrees}도 회전")

    def signal_ready_for_data(self):
        """데이터 수집 준비 완료 신호"""
        self.waiting_for_data = True
        
        # 로봇 준비 완료 신호 전송
        ready_msg = Bool()
        ready_msg.data = True
        self.robot_ready_pub.publish(ready_msg)
        
        # 상태 알림
        status_msg = String()
        # publish cycle and position so collector can group samples per position across cycles
        status_msg.data = f"ready_for_data_{self.cycle_index}_{self.sequence_count}"
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
                                         f"진행={self.sequence_count}/{self.positions_per_rotation}")
    
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
                            print(f"   진행상황: {robot_controller.sequence_count}/{robot_controller.positions_per_rotation}")
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
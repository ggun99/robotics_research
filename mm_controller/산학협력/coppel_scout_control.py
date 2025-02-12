import sys
import os
from scipy.signal import butter, lfilter
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
import numpy as np
from zmqRemoteApi import RemoteAPIClient
from cubic_bezier_ import Bezier
from backstepping import Backstepping



class Scout_Control():
    def __init__(self):
        self.t_robot = 0
        self.r = 0.165
        self.l = 0.582
        print('Program started')
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)
        self.sim.startSimulation()
        self.dummy_position = self.sim.getObjectHandle('/Dummy')
        self.planning_dummy_position = self.sim.getObjectHandle('/Dummy[1]')
        # scout_position = sim.getObjectHandle('/Dummy')
        self.control_points = np.zeros((3, 2))
        self.f_L = self.sim.getObject('/base_link_respondable/front_left_wheel')
        self.f_R = self.sim.getObject('/base_link_respondable/front_right_wheel')
        self.r_L = self.sim.getObject('/base_link_respondable/rear_left_wheel')
        self.r_R = self.sim.getObject('/base_link_respondable/rear_right_wheel')
        self.scout_base = self.sim.getObject('/base_link_respondable')
        self.pose_scout = self.sim.getObjectPosition(self.scout_base, -1)
        self.pose_dummy = self.sim.getObjectPosition(self.dummy_position, -1)
        self.planning_pose_dummy = self.sim.getObjectPosition(self.planning_dummy_position, -1)
        self.x_d = 0
        self.y_d = 0
        self.Bezier = Bezier()
        self.Backstepping = Backstepping()
        value_locked = False  # 값이 고정되었는지 여부

    def scout_control(self, y_e, t_robot, x_desired, y_desired):
        print(self.control_points)
        self.x_d, self.y_d, x_dp, y_dp, x_dpp, y_dpp = self.Bezier.bezier_curve(control_points=self.control_points, t_robot=t_robot)

        vc, wc = self.Backstepping.backstepping_control(self.x_d, self.y_d, x_desired, y_desired, x_dp, y_dp, x_dpp, y_dpp, y_e, K1=12, K2=3, K3=3)

        vc = float(vc)
        if vc > 0.22:
            vc = 0.22
        elif vc < -0.22:
            vc = -0.22
        wc = float(wc)
        if wc > 0.2:
            wc = 0.2
        elif wc < -0.2:
            wc = -0.2
        print(vc,wc)
        return vc, wc

    def butter_lowpass(self, cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        # 필터 적용 함수
    def butter_lowpass_filter(self, data, cutoff, fs, order=10):
            b, a = self.butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

    def moveToAngle_R(self, w_R):
        vel = w_R
        # sim.setJointTargetVelocity(f_L, vel)
        self.sim.setJointTargetVelocity(self.f_R, -vel)
        # sim.setJointTargetVelocity(r_L, vel)
        self.sim.setJointTargetVelocity(self.r_R, -vel)

    def moveToAngle_L(self, w_L):
        vel = w_L
        self.sim.setJointTargetVelocity(self.f_L, vel)
        # sim.setJointTargetVelocity(f_R, -vel)
        self.sim.setJointTargetVelocity(self.r_L, vel)
        # sim.setJointTargetVelocity(r_R, -vel)

    def main(self):
        try:
            while (t := self.sim.getSimulationTime()) < 12:
                s = f'Simulation time: {t:.2f} [s]'
                print(s)
                
                x_desired = self.pose_dummy[0]
                y_desired = self.pose_dummy[1]
                self.control_points[0][0] = self.pose_scout[0]
                self.control_points[0][1] = self.pose_scout[1]
                self.control_points[1][0] = x_desired*3/5
                self.control_points[1][1] = y_desired*4/5
                self.control_points[2][0] = x_desired
                self.control_points[2][1] = y_desired
                
                print(f'Scout pose: {self.pose_scout}')
                y_e = np.arctan((self.pose_scout[1] - self.pose_dummy[1])/(self.pose_scout[0] - self.pose_dummy[0]))
                vc, wc = self.scout_control(y_e, t, x_desired, y_desired)
                
                w_R = vc/self.r + self.l*wc/(2*self.r)
                w_L = vc/self.r - self.l*wc/(2*self.r) # Convert linear velocity to angular velocity
                self.moveToAngle_R(w_R)
                self.moveToAngle_L(w_L)
                new_position = [self.x_d, self.y_d, self.planning_pose_dummy[2]]
                self.sim.setObjectPosition(self.planning_dummy_position, -1, new_position)
                self.client.step()  # Advance simulation by one step

        except Exception as e:
            print(f"Error during simulation step: {e}")
        finally:
            self.sim.stopSimulation()

if __name__ == '__main__':
    scout = Scout_Control()
    scout.main()    

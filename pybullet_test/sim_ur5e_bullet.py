import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import pybullet as p
import pybullet_data


class UR5eBullet:

    def __init__(self, mode="gui") -> None:
        # connect
        if mode == "gui":
            p.connect(p.GUI)
            # p.connect(p.SHARED_MEMORY_GUI)
        if mode == "no_gui":
            p.connect(p.DIRECT)  # for non-graphical
            # p.connect(p.SHARED_MEMORY)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # load model and properties
        self.load_model()
        self.numJoints = self.get_num_joints()
        self.jointNames = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.jointIDs = [1, 2, 3, 4, 5, 6]
        self.gripperlinkid = 9

        # inverse kinematic
        self.lower_limits = [-np.pi] * 6
        self.upper_limits = [np.pi] * 6
        self.joint_ranges = [2 * np.pi] * 6
        self.rest_poses = [0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.joint_damp = [0.01] * 6

    def load_model(self):
        self.planeID = p.loadURDF("plane.urdf", [0, 0, 0])
        self.ur5eID = p.loadURDF("/home/airlab/robotics_research/pybullet_test/ur5e_extract_calibrated.urdf", [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

    def get_visualizer_camera(self):
        (
            width,
            height,
            viewMatrix,
            projectionMatrix,
            cameraUp,
            cameraForward,
            horizontal,
            vertical,
            yaw,
            pitch,
            dist,
            target,
        ) = p.getDebugVisualizerCamera()

        print(f"> width: {width}")
        print(f"> height: {height}")
        print(f"> viewMatrix: {viewMatrix}")
        print(f"> projectionMatrix: {projectionMatrix}")
        print(f"> cameraUp: {cameraUp}")
        print(f"> cameraForward: {cameraForward}")
        print(f"> horizontal: {horizontal}")
        print(f"> vertical: {vertical}")
        print(f"> yaw: {yaw}")
        print(f"> pitch: {pitch}")
        print(f"> dist: {dist}")
        print(f"> target: {target}")

    def set_visualizer_camera(self, cameraDistance=3, cameraYaw=30, cameraPitch=52, cameraTargetPosition=[0, 0, 0]):
        p.resetDebugVisualizerCamera(cameraDistance=cameraDistance, cameraYaw=cameraYaw, cameraPitch=cameraPitch, cameraTargetPosition=cameraTargetPosition)

    def get_num_joints(self):
        return p.getNumJoints(self.ur5eID)

    def get_joint_link_info(self):
        for i in range(self.numJoints):
            (
                jointIndex,
                jointName,
                jointType,
                qIndex,
                uIndex,
                flags,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                linkName,
                jointAxis,
                parentFramePos,
                parentFrameOrn,
                parentIndex,
            ) = p.getJointInfo(self.ur5eID, i)

            print(f"> ---------------------------------------------<")
            print(f"> jointIndex: {jointIndex}")
            print(f"> jointName: {jointName}")
            print(f"> jointType: {jointType}")
            print(f"> qIndex: {qIndex}")
            print(f"> uIndex: {uIndex}")
            print(f"> flags: {flags}")
            print(f"> jointDamping: {jointDamping}")
            print(f"> jointFriction: {jointFriction}")
            print(f"> jointLowerLimit: {jointLowerLimit}")
            print(f"> jointUpperLimit: {jointUpperLimit}")
            print(f"> jointMaxForce: {jointMaxForce}")
            print(f"> jointMaxVelocity: {jointMaxVelocity}")
            print(f"> linkName: {linkName}")
            print(f"> jointAxis: {jointAxis}")
            print(f"> parentFramePos: {parentFramePos}")
            print(f"> parentFrameOrn: {parentFrameOrn}")
            print(f"> parentIndex: {parentIndex}")

    def control_single_motor(self, jointIndex, jointPosition, jointVelocity=0):
        p.setJointMotorControl2(
            bodyIndex=self.ur5eID,
            jointIndex=jointIndex,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPosition,
            targetVelocity=jointVelocity,
            positionGain=0.03,
        )

    def control_array_motors(self, jointPositions, jointVelocities=[0, 0, 0, 0, 0, 0]):
        p.setJointMotorControlArray(
            bodyIndex=self.ur5eID,
            jointIndices=self.jointIDs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=jointPositions,
            targetVelocities=jointVelocities,
            positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        )

    def get_single_joint_state(self):
        jointPosition, jointVelocity, jointReactionForce, appliedJointMotorTorque = p.getJointState(self.ur5eID, jointIndex=1)
        return jointPosition, jointVelocity, jointReactionForce, appliedJointMotorTorque

    def get_array_joint_state(self):
        j1, j2, j3, j4, j5, j6 = p.getJointStates(self.ur5eID, jointIndices=self.jointIDs)
        return j1, j2, j3, j4, j5, j6

    def get_array_joint_positions(self):
        j1, j2, j3, j4, j5, j6 = self.get_array_joint_state()
        return (j1[0], j2[0], j3[0], j4[0], j5[0], j6[0])

    def forward_kin(self):
        (
            link_trn,
            link_rot,
            com_trn,
            com_rot,
            frame_pos,
            frame_rot,
            link_vt,
            link_vr,
        ) = p.getLinkState(self.ur5eID, self.gripperlinkid, computeLinkVelocity=True, computeForwardKinematics=True)
        return link_trn, link_rot

    def inverse_kin(self, positions, quaternions):
        joint_angles = p.calculateInverseKinematics(
            self.ur5eID,
            self.gripperlinkid,
            positions,
            quaternions,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            jointDamping=self.joint_damp,
            restPoses=self.rest_poses,
        )
        return joint_angles

    def contact_point(self):
        contact_points = p.getContactPoints(bodyA=self.ur5eID, bodyB=self.tableID)
        print(f"> contact_points: {contact_points}")
        # for point in contact_points:
        #     print(f"Contact point details: {point}")

    def closest_point(self):
        closest_points = p.getClosestPoints(bodyA=self.ur5eID, bodyB=self.tableID, distance=0.5)
        print(f"> closest_points: {closest_points}")

    def reset_array_joint_state(self, targetValues):
        for i in range(6):
            p.resetJointState(self.ur5eID, jointIndex=self.jointIDs[i], targetValue=targetValues[i])

    def collisioncheck(self):
        p.performCollisionDetection()

    def compute_jac(self, joint_position):
        com_trn = [0.0, 0.0, 0.0] # assume to be zeros for now
        zero_vec = [0.0] * 6
        jac_t, jac_r = p.calculateJacobian(self.ur5eID, self.gripperlinkid, com_trn, joint_position, zero_vec, zero_vec)
        return jac_t, jac_r


if __name__ == "__main__":
    robot = UR5eBullet("gui")

    try:
        while True:
            joint_position=[1.3940227031707764, -1.5853477917113246, 1.7367637793170374, -0.07264550149951177, -0.18519813219179326, 0.038892269134521484]
            jac_t, jac_r = robot.compute_jac(joint_position)
            jt = np.array(jac_t)
            print(f"==>> jt.shape: {jt.shape}")
            jr = np.array(jac_r)
            print(f"==>> jr.shape: {jr.shape}")


            qKey = ord("q")
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                break

            p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()

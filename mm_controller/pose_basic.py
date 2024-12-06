import numpy as np


class DifferentialDrivePoseBasicController:
    """
    [Summary] : Pose Basic Controller provide control input toward desired position until distance << eta, then performs orientation correction.

    """

    def __init__(self, robot) -> None:
        self.robot = robot

        # controller constants for tuning
        self.K1 = 0.8
        self.K2 = 5
        self.Korient = 0.5

        # correction
        self.dTol = 0.01  # Tolerance distance (to the intermediate point) for switch
        self.state = False  # State: 0 - go to position, 1 - go to orientation

    def kinematic_control(self, currentPose, referencePose):
        # controller switch condition
        if np.linalg.norm([(referencePose[0, 0] - currentPose[0, 0]), (referencePose[1, 0] - currentPose[1, 0])]) < self.dTol:
            self.state = True

        # position controller
        if not self.state:
            phiref = np.arctan2((referencePose[1, 0] - currentPose[1, 0]), (referencePose[0, 0] - currentPose[0, 0]))
            qRef = np.array([referencePose[0, 0], referencePose[1, 0], phiref]).reshape(3, 1)
            e = qRef - currentPose
            vc = self.K1 * np.sqrt((e[0, 0] ** 2) + (e[1, 0] ** 2))
            wc = self.K2 * e[2, 0]

        # orientation controller
        if self.state:
            e = referencePose[2, 0] - currentPose[2, 0]
            vc = 0
            wc = self.Korient * e

        # physical limit
        if abs(vc) > 0.8:
            vc = 0.8 * np.sign(vc)

        return np.array([[vc], [wc]])


class DifferentialDrivePositionForwardController:
    """
    [Summary] : Position Basic Controller provide only control input toward desired position until distance.

    """

    def __init__(self, robot) -> None:
        self.robot = robot

        # controller constants for tuning
        self.K1 = 0.1
        self.K2 = 0.5

    def kinematic_control(self, currentPose, referencePose):
        e = referencePose - currentPose

        # position controller
        vc = self.K1 * np.sqrt((e[0, 0] ** 2) + (e[1, 0] ** 2))
        wc = self.K2 * e[2, 0]

        # physical limit
        if abs(vc) > 0.8:
            vc = 0.8 * np.sign(vc)

        return np.array([vc, wc]).reshape(2, 1)


if __name__ == "__main__":
    import os
    import sys

    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    import matplotlib.pyplot as plt
    from robot.mobile.differential import DifferentialDrive
    from simulator.integrator_euler import EulerNumericalIntegrator

    # create robot and controller
    robot = DifferentialDrive(wheelRadius=0, baseLength=0.3, baseWidth=0.3)
    controller = DifferentialDrivePoseBasicController(robot=robot)

    # simulator
    def dynamic(currentPose, input):
        return robot.forward_external_kinematic(input, currentPose[2, 0])

    def desired(currentPose, time):
        return np.array([4.0, 4.0, 0]).reshape(3, 1)

    def control(currentPose, desiredPose):
        return controller.kinematic_control(currentPose, desiredPose)

    q0 = np.array([1.0, 0.0, -np.pi]).reshape(3, 1)
    tSpan = (0, 15)
    dt = 0.01
    intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
    timeSteps, states, desireds, controls = intg.simulation()

    plt.plot(states[0, :], states[1, :])
    plt.grid(True)
    plt.show()

    plt.plot(timeSteps, states[0, :])
    plt.plot(timeSteps, states[1, :])
    plt.plot(timeSteps, states[2, :])
    plt.grid(True)
    plt.show()

    plt.plot(timeSteps, desireds[0, :])
    plt.plot(timeSteps, desireds[1, :])
    plt.plot(timeSteps, desireds[2, :])
    plt.grid(True)
    plt.show()

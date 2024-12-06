import sys
import os

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
import numpy as np
from zmqRemoteApi import RemoteAPIClient

print('Program started')
client = RemoteAPIClient()
sim = client.getObject('sim')
simIK = client.getObject('simIK')

client.step()
sim.startSimulation()


def sysCall_init():
    global ikEnv
    global ikGroup_damped
    simBase = sim.getObject('/UR5')
    simTip = sim.getObject('./Tip')
    simTarget = sim.getObjectHandle('/Dummy')
    f_L = sim.getObject('/base_link_respondable/front_left_wheel')
    f_R = sim.getObject('/base_link_respondable/front_right_wheel')
    r_L = sim.getObject('/base_link_respondable/rear_left_wheel')
    r_R = sim.getObject('/base_link_respondable/rear_right_wheel')
    sim.setJointTargetVelocity(f_L, 0)
    sim.setJointTargetVelocity(f_R, 0)
    sim.setJointTargetVelocity(r_L, 0)
    sim.setJointTargetVelocity(r_R, 0)
    
    # Create an IK environment
    ikEnv =simIK.createEnvironment()

    # Create an IK group
    ikGroup_damped = simIK.createIkGroup(ikEnv)

    simIK.setIkGroupCalculation(ikEnv, ikGroup_damped, simIK.method_damped_least_squares, 0.24, 99)
    simIK.addIkElementFromScene(ikEnv, ikGroup_damped, simBase, simTip, simTarget, simIK.constraint_pose)

def sysCall_actuation():
    simIK.applyIkEnvironmentToScene(ikEnv, ikGroup_damped)


def sysCall_cleanup():
    simIK.eraseEnvironment(ikEnv)

sysCall_init()

while (t := sim.getSimulationTime()) < 10:
    sysCall_actuation()

sysCall_cleanup()
sim.stopSimulation()
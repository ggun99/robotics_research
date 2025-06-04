import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math

np.set_printoptions(precision=3, suppress=True, linewidth=120)
env = swift.Swift()
env.launch(realtime=True)

ax_goal = sg.Axes(0.1)
env.add(ax_goal)

frankie = rtb.models.UR5()
frankie.qdlim = np.array([2.17]*6)
frankie.q = np.array([0., -np.pi/2, np.pi/2, 0., np.pi/2, 0.])
print(frankie.q)
env.add(frankie)

arrived = False
dt = 0.025


def step_robot(r: rtb.ERobot, Tep):

    wTe = r.fkine(r.q)
    print(r.q)
    eTep = np.linalg.inv(wTe) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))
    print(et)
    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 6)
    print('r.n' , r.n)

    # Joint velocity component of Q
    Q[: r.n, : r.n] *= Y
    Q[:2, :2] *= 1.0 / et

    # Slack component of Q
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)
    # print(Q)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)

    v[3:] *= 1.3

    # The equality contraints
    robjac = r.jacobe(r.q)
    print("==========================")
    # print(robjac.shape)

    Aeq = np.c_[robjac, np.eye(6)]
    # print('Aeq.shape', Aeq.shape)
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.1

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (np.zeros(2), -r.jacobm(start=r.links[4]).reshape((r.n - 2,)), np.zeros(6))
    )
    
    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε
    # print('c', c)
    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
    qd = qd[: r.n]
    print("qd:", qd)
    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    if et < 0.02:
        return True, qd
    else:
        return False, qd

# Behind
env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
wTep = frankie.fkine(frankie.q) #* sm.SE3.Rz(np.pi) #* sm.SE3.Rx(-np.pi/2)
# wTep.A[:3, :3] = np.diag([1, 1, 1])
wTep.A[0, -1] -= 0.5
wTep.A[2, -1] -= 0.25
ax_goal.T = wTep
env.step()


while not arrived:

    arrived, frankie.qd = step_robot(frankie, wTep.A)
    env.step(dt)
    # print('q : ', frankie.q)
    # print('_q : ', frankie._q)
    # print('links : ', frankie.links[8])
    # Reset bases
    base_new = frankie.fkine(frankie._q)
    # frankie._T = base_new.A
    # frankie.q[:2] = 0

env.hold()
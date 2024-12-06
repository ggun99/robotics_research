import gjk
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

# color
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE  = (  0,   0, 255)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)

# information of test environment
robot_link1 = 150
robot_link2 = 100
robot_thickness = 20
obstacle = ((50, 150), 30)
# obstacle = ((np.random.randint(10, 200),np.random.randint(10, 200)), np.random.randint(10, 50))

def file_clear():

    # File clear
    file_path = "/home/jeongil/collision/making_file/result/2dof_2D_collision_data.txt"
    with open(file_path, "w") as file:          
        file.write(f"")
    file_path = "/home/jeongil/collision/making_file/result/2dof_2D_graph_data.txt"
    with open(file_path, "w") as file: 
        file.write(f"")
    file_path = "/home/jeongil/collision/making_file/result/2dof_2D_input.txt"
    with open(file_path, "w") as file:          
        file.write(f"")

# calculate collision with circle obstacle
def run_all_angle():

    start = time.time()

    for q1_rad in range(0, 360):
        for q2_rad in range(0, 360):
            
            q1 = math.radians(q1_rad)
            q2 = math.radians(q2_rad)

            link1_1=np.array(([0, robot_thickness/2], [0,-robot_thickness/2], [robot_link1, -robot_thickness/2], [robot_link1, robot_thickness/2]))
            r1 = np.array(([math.cos(q1), -math.sin(q1)], [math.sin(q1), math.cos(q1)]))
            link1_ro = np.matmul(r1, link1_1.T)
            link_1 = (link1_ro.T)



            link2_1 = np.array(([0, robot_thickness/2], [0, -robot_thickness/2], [robot_link2, -robot_thickness/2], [robot_link2, robot_thickness/2]))
            r2 = np.array(([math.cos(q2), -math.sin(q2)], [math.sin(q2), math.cos(q2)]))
            link2_ro = np.matmul(r2, link2_1.T)
            link2_ro += np.array(([robot_link1], [0]))
            link2_ro2 = np.matmul(r1, link2_ro)
            link_2 = (link2_ro2.T)

            # collision check with circle obstacle
            collide_1 = gjk.collidePolyCircle(link_1, obstacle)
            circle(obstacle)
            collide_2 = gjk.collidePolyCircle(link_2, obstacle)
            circle(obstacle)

            # collision true = 0
            if collide_1 or collide_2:
                collision = 0
            else:
                collision = 1
                 
            # # show robot link
            # fig, ax = plt.subplots()
            # body = Polygon(link_1)
            # ax.add_patch(body)
            # body = Polygon(link_2)
            # ax.add_patch(body)
            # ax.set_xlim(-300, 300)
            # ax.set_ylim(-300, 300)
            # plt.show()

            # save result in txt file and  recalculate
            file_path = "/home/jeongil/collision/making_file/result/2dof_2D_collision_data.txt"
            if (collide_1 or collide_2):
                with open(file_path, "a") as file:
                    file.write(f"collision  q1 : {q1_rad}   q2 : {q2_rad}\n")
                
            file_path = "/home/jeongil/collision/making_file/result/2dof_2D_graph_data.txt"
            with open(file_path, "a") as file:          
                file.write(f"{q1_rad}, {q2_rad}, {'0' if (collide_1 or collide_2) else '1'}\n")

    file_path = "/home/jeongil/collision/making_file/result/2dof_2D_input.txt"
    with open(file_path, "a") as file:          
        file.write(f"robot_link1 : {robot_link1}\nrobot_link2 : {robot_link2}\n(robot_thickness/2) : {(robot_thickness/2)}\nobstacle : {obstacle}")

    end = time.time()
    
    print(f"{end - start :.5f} sec")

    # plt.xlabel("joint 1 angle(q1, degrees)")
    # plt.ylabel("joint 2 angle(q2, degrees)")
    # plt.title("C-space")
    # plt.show()

# make a C_space graph and save
def C_space():

    file_path = "/home/jeongil/collision/making_file/result/2dof_2D_graph_data.txt"

    x_values = []
    y_values = []
    z_values = []

    with open(file_path, "r") as file:
        for line in file:
            data = line.strip().split(",")
            x = float(data[0])
            y = float(data[1])
            z = float(data[2])
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)

    x_unique = np.sort(np.unique(x_values))
    y_unique = np.sort(np.unique(y_values))
    z_array = np.zeros((len(y_unique), len(x_unique)))

    for x, y, z in zip(x_values, y_values, z_values):
        x_index = np.where(x_unique == x)[0][0]
        y_index = np.where(y_unique == y)[0][0]
        z_array[y_index, x_index] = z

    plt.imshow(z_array, cmap='Pastel1', origin='upper', extent=[x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]])

    plt.xlabel("joint 1 angle(q1, degrees)")
    plt.ylabel("joint 2 angle(q2, degrees)")
    plt.title("C-space")
    plt.savefig('/home/jeongil/collision/making_file/result/2dof_2D_C-space.png')
    plt.show()

def pairs(points):
    for i, j in enumerate(range(-1, len(points) - 1)):
        yield (points[i], points[j])
def circles(cs, color=BLACK, camera=(0, 0)):
    for c in cs:
        circle(c, color, camera)
def circle(c, color=BLACK, camera=(0, 0)):
    ()
def polygon(points, color=BLACK, camera=(0, 0)):
    for a, b in pairs(points):
        line(a, b, color, camera)
def line(start, end, color=BLACK, camera=(0, 0)):
    ()
def add(p1, p2):
    return p1[0] + p2[0], p1[1] + p2[1]

# run code 
if __name__ == '__main__':

    file_clear()
    run_all_angle()
    C_space()



# 로봇팔 위치 계산법 변경
    # 기구학(forward kinematic(로봇공학)) 이용
# 결과값이 이상하게 나온다
# joint2값이 180도 증가한 결과값이 출력
# (실제 정답 : 100도 --> 파일 결과 : 280도 그래프 출력)
# joint1값은 문제없음
# 이유 못찾음...
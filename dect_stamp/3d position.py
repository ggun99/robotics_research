import numpy as np
from position import Position
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = []
Y = []
Z = []



def cal_mean(position):
    pose = position
    sumvalue_z = np.sum(pose[:,2])
    sumvalue_x = np.sum(pose[:,0])
    sumvalue_y = np.sum(pose[:,1])
    # print('std_x',np.std(pose[:,0]))
    # print('mean_x',sumvalue_x/100)
    # print('std_y',np.std(pose[:,1]))
    # print('mean_y',sumvalue_y/100)
    # print('std_z',np.std(pose[:,2]))
    # print('mean_z',sumvalue_z/100)
    # print()
    X.append(sumvalue_x/100)
    Y.append(sumvalue_y/100)
    Z.append(sumvalue_z/100)
    

def cal_length(num1, num2):
    point1 = np.array([X[num1], Y[num1], Z[num1]])
    point2 = np.array([X[num2], Y[num2], Z[num2]])
    distance = np.linalg.norm(point2 - point1)
    #print(f"Distance between points {num1} and {num2}: {distance}")
    return distance
    
position_ball = Position()


for i in range(1,28):
    point_name = f"point{i}"
    point_value = getattr(position_ball, point_name)
    #print(i)
    cal_mean(point_value)
x_pairs = [(4, 7),(1, 4),(3, 6),(2, 5),(5, 8),(6, 9),(13, 16),(10, 13),(11, 14),(14, 17),(15, 18),(12, 15),(25,22),(22,19),(20,23),(23,26),(21,24),(24,27)]
y_pairs = [(1, 2), (2, 3),(5, 6),(4, 5),(7, 8),(8, 9),(11, 12),(10, 11),(13, 14),(14, 15),(17, 18),(16, 17),(19,20),(20,21),(22,23),(23,24),(27,26),(25,26)]
z_pairs = [(10, 19), (1, 10), (2, 11),(11, 20),(3, 12),(12, 21),(6, 15),(15, 24),(18,27),(8,17),(17,26),(7,16),(16,25),(4,13),(13,22),(9, 18),(5,14),(14,23)]

x_distances = []
y_distances = []
z_distances = []

for pair in x_pairs:
    distance = cal_length(pair[0]-1, pair[1]-1)
    x_distances.append(distance)
for pair in y_pairs:
    distance = cal_length(pair[0]-1, pair[1]-1)
    y_distances.append(distance)
for pair in z_pairs:
    distance = cal_length(pair[0]-1, pair[1]-1)
    z_distances.append(distance)

print(f'x_len: {len(x_distances)}, x_mean: {np.mean(x_distances)}, x_std: {np.std(x_distances)}')
print(f'y_len: {len(y_distances)}, y_mean: {np.mean(y_distances)}, y_std: {np.std(y_distances)}')
print(f'z_len: {len(z_distances)}, z_mean: {np.mean(z_distances)}, z_std: {np.std(z_distances)}')

selected_pairs = [(1, 2), (2, 3), (3, 6),(5, 6),(4, 5),(4, 7),(7, 8),(8, 9),(1, 4),(2, 5),(5, 8),(6, 9),
                  (9, 18),(17, 18),(16, 17),(13, 16),(13, 14),(14, 15),(12, 15),(11, 12),(10, 11),(10, 13),(11, 14),(14, 17),(15, 18),
                  (10, 19), (1, 10), (2, 11),(11, 20),(3, 12),(12, 21),(6, 15),(15, 24),(18,27),(8,17),(17,26),(7,16),(16,25),(4,13),(13,22),(5,14),(14,23),
                  (25,22),(22,19),(19,20),(20,21),(21,24),(24,27),(27,26),(25,26),(22,23),(20,23),(23,26),(23,24)]
distances = []
# print(len(selected_pairs))
for pair in selected_pairs:
    distance = cal_length(pair[0]-1, pair[1]-1)
    distances.append(distance)

# 3차원 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='b', marker='o')

for i, pair in enumerate(selected_pairs):
    x = [X[pair[0]-1], X[pair[1]-1]]
    y = [Y[pair[0]-1], Y[pair[1]-1]]
    z = [Z[pair[0]-1], Z[pair[1]-1]]
    
    ax.plot(x, y, z, label=f'Distance: {distances[i]:.2f}', marker='o')

    # 선분 중간 지점 계산
    mid_x = (x[0] + x[1]) / 2
    mid_y = (y[0] + y[1]) / 2
    mid_z = (z[0] + z[1]) / 2
    
    # 거리 정보 텍스트 추가
    ax.text(mid_x, mid_y, mid_z, f'{distances[i]:.5f}', fontsize=8, color='black', ha='right')

#print(distances)

# 축 레이블 설정
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
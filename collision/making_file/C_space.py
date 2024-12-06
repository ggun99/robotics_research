import matplotlib.pyplot as plt
import ast

file_path = "/home/jeongil/collision/making_file/result/2dof_2D_graph_data.txt"

coordinates_0 = []
coordinates_1 = []

with open(file_path, "r") as file:
    for line in file:
        data = ast.literal_eval(line.strip())
        coordinates = data[0]
        value = data[1]
        if value == 0:
            coordinates_0.append(coordinates)
        elif value == 1:
            coordinates_1.append(coordinates)

for coordinates in coordinates_0:
    x, y = coordinates
    plt.scatter(x, y, color='red', marker='o', label='Value 0')

for coordinates in coordinates_1:
    x, y = coordinates
    plt.scatter(x, y, color='blue', marker='x', label='Value 1')

plt.xlabel("joint 1 angle(q1, degrees)")
plt.ylabel("joint 2 angle(q2, degrees)")
plt.title("C-space")
plt.show()
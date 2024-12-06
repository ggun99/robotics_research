import numpy as np
import random
from collections import deque

# def generate_random_coordinates(x_min, x_max, y_min, y_max):
#   """
#   Generates a random (x, y) coordinate within the specified range, returning integers.

#   Args:
#       x_min: Minimum value for the x-coordinate (inclusive).
#       x_max: Maximum value for the x-coordinate (inclusive).
#       y_min: Minimum value for the y-coordinate (inclusive).
#       y_max: Maximum value for the y-coordinate (inclusive).

#   Returns:
#       A tuple containing the randomly generated (x, y) coordinate as integers.
#   """
#   x = random.randint(x_min, x_max)
#   y = random.randint(y_min, y_max)

#   return (x, y)

# data_list = deque([])

# def mean_window(data,length):

#     # x_sum = 0  # Initialize sum for x coordinates
#     # y_sum = 0  # Initialize sum for y coordinates

#     x = data[0][0]
#     y = data[0][1]
#     data_list.append([x,y])

#     for i in range(len(data_list)):
#         x_, y_ = data_list[i][0], data_list[i][1]  # Unpack the tuple
#         x_ += x_
#         y_ += y_

#         x_m = x_/length  # Calculate mean of x coordinates
#         y_m = y_/length 
  
#     if len(data_list) > length:
#         data_list.popleft()
#     x_m = x_m
#     y_m = y_m

#     return x_m, y_m


# while True:
#     data_ = []
#     x_min = 0
#     x_max = 10
#     y_min = 0
#     y_max = 10
#     x, y = generate_random_coordinates(x_min, x_max, y_min, y_max) #data
#     # print(f"Random coordinate: ({x}, {y})")
#     data_.append([x,y])
#     xm, ym = mean_window(data_,5)
#     print('aa',xm)


def mean_window(camera_location,data,length):
    if camera_location == 'r':
        data_list = data_list_r
    if camera_location == 'l':
        data_list = data_list_l
    else :
        print('camera location has only r or l')

    x = data[0]
    y = data[1]
    
    if x + y != 0 and (x or y is None):
        data_list.append([x,y])
    
        if len(data_list) > length-1:
            data_list.popleft()
        x_sum = 0 
        y_sum = 0 
        for i in range(len(data_list)):
            x_, y_ = data_list[i][0], data_list[i][1]  # Unpack the tuple
            x_sum += x_
            y_sum += y_
        x_m = x_sum/len(data_list)  # Calculate mean of x coordinates
        y_m = y_sum/len(data_list)
    else : 
        x_sum = 0 
        y_sum = 0
        for i in range(len(data_list)):
            x_, y_ = data_list[i][0], data_list[i][1]  # Unpack the tuple
            x_sum += x_
            y_sum += y_
        if len(data_list) != 0:
            x_m = x_sum/len(data_list)  # Calculate mean of x coordinates
            y_m = y_sum/len(data_list) 
        else:
            x_m = x_sum/1  # Calculate mean of x coordinates
            y_m = y_sum/1

    return int(x_m), int(y_m)


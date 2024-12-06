"""
========================================================
Check the Collision between Links and obstacle with GJK
========================================================
"""
print(__doc__)
import time
import numpy as np
from pytransform3d.transformations import transform_from
import pytransform3d.rotations as pyrot
from distance3d import gjk, colliders
import math
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import itertools
import random

L1_len = 10
L2_len = 5
L3_len = 3
radius = 1

collision = []
no_collision = []
result = []
theta_data = []

# def rand_choice(data1, data2):
#     # 랜덤한 인덱스 생성
#     random_indices = np.random.choice(data1.shape[0], 1000, replace=False)

#     # 랜덤한 인덱스에 해당하는 행 추출
#     random_data1 = data1[random_indices]
#     random_data2 = data2[random_indices]
#     # random_data3 = data3[random_indices]
#     return random_data1, random_data2 #, random_data3
def theta_list(theta_data):
    for i in range(36):
        theta = math.radians(i*10)
        # print(f"==>> theta: {theta}")
        for j in range(36):
            theta2 = math.radians(j*10)
            theta_data.append([theta,theta2])
    rand_theta_data = random.sample(theta_data,1000)
    return rand_theta_data
def Arm_with_Mobile(theta, theta2):
    center1 = np.array([0,0,L1_len])
    # Link1
    p=np.array([0.,0.,L1_len/2])
    a = np.array([0.0, 0.0, 1.0, np.degrees(theta)])
    R_1 = pyrot.matrix_from_axis_angle(a)
    L1_2_origin = transform_from(R_1,p)
    # Link2
    p2=np.array([L2_len/2*np.sin(theta2),0.,L1_len+L2_len/2*np.cos(theta2)])
    a2=np.array([0.,1.,0.,np.degrees(theta2)])
    R_2 = pyrot.matrix_from_axis_angle(a2)
    L2_2_L1 = transform_from(R_2@R_1,p2)
   
    # Links
    L1 = colliders.Cylinder(L1_2_origin, radius, L1_len)
    L2 = colliders.Cylinder(L2_2_L1, radius, L2_len)
    
    # Joints
    J1 = colliders.Sphere(center1, radius)
    center2 = np.array([L2_len*np.sin(theta2),0.,L1_len+L2_len*np.cos(theta2)])
    J2 = colliders.Sphere(center2, radius)
    
    # mobile robot
    mobile_box_size = np.array([5.,10.,3.])
    p_m = np.array([1.,0.,-1.5])
    a_m = np.array([0.,0.,1.,theta])
    mobile_2_origin = transform_from(pyrot.matrix_from_axis_angle(a_m),p_m)
    # mobile = colliders.Box(mobile_box,mobile_box_size)
    p_mw = np.array([-3.,0.,-3.3])
    p_mw2 = np.array([5.,0.,-3.3])
    a_mw = a_m = np.array([1.,0.,0.,theta])
    mobile_wheel_2_box = transform_from(pyrot.matrix_from_axis_angle(a_mw),p_mw)
    mobile_wheel_2_box2 = transform_from(pyrot.matrix_from_axis_angle(a_mw),p_mw2)

    return L1, L2

def collision_check(theta_data):
    dataset = []
    accumulated_time = 0.0
    
    start = time.time()
    print('collision checking..')
    for l in range(len(theta_data)):
        theta = theta_data[l][0]
        theta2 = theta_data[l][1]
        # theta3 = theta_data[l][2]

        # obstacle(sphere)
        obs_p = np.array([5., 2., 10.])
        obs_r = 3
        obs = colliders.Sphere(obs_p, obs_r)

        # if you want to know distance with link2 and obstacle
        #dist, closest_point_cylinder, closest_point_cylinder2, _ = gjk.gjk(L2, obs)

        # collision check
        # s = np.degrees(theta)
        # d = np.degrees(theta2)
        # f = np.degrees(theta3)
        L1 ,L2 = Arm_with_Mobile(theta, theta2)
        # check_collision_L3 = gjk.gjk_intersection(L3,obs)
      
        check_collision_L2 = gjk.gjk_intersection(L2,obs)
        if check_collision_L2 is True:
            dataset.append([theta,theta2,1])
            # collision.append([s,d,f])
            # result.append(1)
        else:
            check_collision_L1 = gjk.gjk_intersection(L1,obs)
            if check_collision_L1 is True:
                dataset.append([theta,theta2,1])
                # collision.append([s,d,f])
                # result.append(1)
            else:  
                dataset.append([theta,theta2,-1])
                # no_collision.append([s,d,f])
                # result.append(-1)

    end = time.time()
    accumulated_time += end - start
    return dataset

def print_score(y_true, y_pred, average='binary'):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    print('accuracy:', acc)
    print('precision:', pre)
    print('recall:', rec)
    print('f1 score:', f1)
    return acc, f1

rand_thetadata = theta_list(theta_data)
dataset = collision_check(rand_thetadata)
# Fastron
# random_dataset = random.sample(dataset, 4000)
rand_dataset = np.array(dataset)
print(f"==>> dataset.shape: {rand_dataset.shape}")
data = rand_dataset[:, 0:2]
y = rand_dataset[:, [2]] 

N = data.shape[0]  # number of datapoint = number of row the dataset has
d = data.shape[1]  # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
g = 10  # kernel width
r = 10  # conditional bias
maxUpdate = 500  # max update iteration
maxSupportPoints = 1500  # max support points
G = np.zeros((N, N), dtype=np.float32)  # kernel gram matrix guassian kernel of dataset
alpha = np.zeros((N, 1), dtype=np.float32)  # weight, init at zero
F = np.zeros((N, 1), dtype=np.float32)  # hypothesis
redund = np.zeros((N, 1), dtype=np.float32)
margin = np.zeros((N, 1), dtype=np.float32)

# active learning parameters
allowance = 800  # number of new samples
kNS = 4  # number of points near supports
gamma = 0.5  # Gaussian sampling std
exploitP = 0.5  # proportion of exploitation samples
gramComputed = [0]*N # np.zeros((N, 1))


def gaussian_kernel(x, y, gamma):
    """Gaussian Kernel measuring similarity between 2 vectors"""
    distance = np.linalg.norm(x - y)
    return np.exp(-gamma * distance**2)


def hypothesis(queryPoint, data, alpha, g):
    term = []
    for i, xi in enumerate(data):
        term.append(alpha[i] * gaussian_kernel(xi, queryPoint, g))
    ypred = np.sum(term) # np.sign(sum(term))  
    return ypred


def eval(queryPoint, data, alpha, gamma):
    term = []
    for i, alphai in enumerate(alpha):
        if alphai != 0.0:
            term.append(alphai * gaussian_kernel(data[i], queryPoint, gamma))
    ypred = np.sign(sum(term))
    return ypred


def compute_kernel_gram_matrix(G, data, gamma):
    for i in range(N):
        for j in range(N):
            print(i,j)
            G[i, j] = gaussian_kernel(data[i], data[j], gamma)

    return G

def all_positive(matrix):
  """
  반복문을 사용하여 행렬의 모든 값이 0보다 큰지 확인합니다.

  Args:
    matrix: 2차원 행렬 (리스트의 리스트)

  Returns:
    모든 값이 0보다 크면 True, 그렇지 않으면 False를 반환합니다.
  """
  for row in matrix:
    for element in row:
      if element <= 0:
        return False
  return True

def keep_select_rows(mat, rows_to_retain):
    """
    Keeps only the specified rows in the matrix and removes the rest.

    Parameters:
    mat (numpy.ndarray): The original matrix.
    rows_to_retain (numpy.ndarray): Array of row indices to retain.

    Returns:
    numpy.ndarray: A new matrix with only the specified rows.
    """
    # Ensure rows_to_retain is a numpy array
    rows_to_retain = np.array(rows_to_retain)
    
    # Create a new matrix with only the retained rows
    new_mat = mat[rows_to_retain, :]
    
    return new_mat


def keep_select_cols(mat, cols_to_retain):
    """
    Keeps only the specified columns in the matrix and removes the rest.

    Parameters:
    mat (numpy.ndarray): The original matrix.
    cols_to_retain (numpy.ndarray): Array of column indices to retain.

    Returns:
    numpy.ndarray: A new matrix with only the specified columns.
    """
    # Ensure cols_to_retain is a numpy array
    cols_to_retain = np.array(cols_to_retain)
    
    # Create a new matrix with only the retained columns
    new_mat = mat[:, cols_to_retain]
    
    return new_mat

def find(alpha):
    """
    Find indices of non-zero elements in the array.
    
    Parameters:
    alpha (numpy.ndarray): Input array.
    
    Returns:
    numpy.ndarray: Indices of non-zero elements.
    """
    return np.nonzero(alpha)[0]

def keep_select_rows_cols(mat, rows_to_retain, cols_to_retain, shift_only):
    """
    Keeps only the specified rows and columns in the matrix and removes the rest.

    Parameters:
    mat (numpy.ndarray): The original matrix.
    rows_to_retain (numpy.ndarray): Array of row indices to retain.
    cols_to_retain (numpy.ndarray): Array of column indices to retain.
    shift_only (bool): If True, only shift elements without resizing.

    Returns:
    numpy.ndarray: A new matrix with only the specified rows and columns.
    """
    # Ensure rows_to_retain and cols_to_retain are numpy arrays
    rows_to_retain = np.array(rows_to_retain)
    cols_to_retain = np.array(cols_to_retain)
    
    # Create a new matrix with only the retained rows and columns
    new_mat = mat[np.ix_(rows_to_retain, cols_to_retain)]
    
    # If shiftOnly is True, just shift elements without resizing
    if shift_only:
        for j in range(len(cols_to_retain)):
            for i in range(len(rows_to_retain)):
                mat[i, j] = mat[rows_to_retain[i], cols_to_retain[j]]
        return mat[:len(rows_to_retain), :len(cols_to_retain)]
    
    # If shiftOnly is False, return the resized matrix
    return new_mat

def sparsify():
        retain_idx = find(alpha)

        N = retain_idx.size
        numberSupportPoints = N

        # Sparsify model
        data = keep_select_rows(data, retain_idx)
        alpha = keep_select_rows(alpha, retain_idx)
        gram_computed = keep_select_rows(gram_computed, retain_idx)
        G = keep_select_rows_cols(G, retain_idx, retain_idx, True)

        # Sparsify arrays needed for updating
        F = keep_select_rows(F, retain_idx)
        y = keep_select_rows(y, retain_idx)

def one_step_weight_kernel_update(alpha, F, data, G, N, g, r, maxUpdate):
    for iter in range(maxUpdate):
        print('iter',iter)
        for i in range(N): 
            redund[i] = y[i]*(F[i]-alpha[i])
        
        for i in range(N):
            j = None
            if redund[i] > 0 and alpha[i] != 0: # Remove redundant support points
                j = np.argmax(redund)
                for l in range(N):
                    F[l] -= G[l, j]*alpha[j]
                    alpha[j] = 0
                    redund[j] = y[j]*(F[j]-alpha[j])
                print('1',i,j)

        for i in range(N):
            margin[i] = y[i] * hypothesis(data[i], data, alpha, g)
        
        if all_positive(margin): # margin-based prioritization
            print('h')
            return alpha, F
        else:
            j = np.argmin(margin)
            print('2',j)
        if y[j]>0: # one-step weight correction with conditional biasing
            delta = r * y[j] - F[j]
            print('3')
        else:
            delta = y[j] - F[j]
            print('4')
        
        alpha[j] += delta
        for k in range(N):
            F[k] += G[k,j]*delta
    return alpha, F

def compute_gram_matrix_col(data, G, gramComputed, idx, start_idx=0, g=1):
    N, d = data.shape
    r2 = np.ones(N - start_idx)
    for j in range(d):
        r2 += g / 2 * np.square(data[start_idx:N, j] - data[idx, j])
    G[start_idx:N, idx] = 1 / (r2 * r2)
    gramComputed[idx] = 1
    # return G, gramComputed

def remove_duplicates_and_sort(nested_list):
  """
  중첩 리스트의 중복된 값을 제거하고 하나의 정렬된 리스트로 만드는 함수입니다.

  Args:
    nested_list: 중첩 리스트입니다.

  Returns:
    중복 제거되고 정렬된 리스트입니다.
  """

  flat_list = list(itertools.chain.from_iterable(nested_list))
  sorted_list = sorted(set(flat_list))
  return sorted_list

## Active Learning ###
def active_learning(data, allowance, kNS, alpha, N, G, gramComputed):
    """
    Implements the active learning strategy for Fastron.

    Args:
        data (np.ndarray): Current dataset (shape: (N, d)).
        N_prev (int): Number of data points before active learning.
        allowance (int): Number of new data points to be added.
        kNS (int): Threshold for exploitation (number of support points to copy).
        alpha (np.ndarray): Lagrangian multipliers (shape: (N,)).
        G (np.ndarray): Gram matrix (shape: (N, N)).
        gramComputed (np.ndarray): Boolean array indicating computed Gram matrix entries (shape: (N,)).
        F (np.ndarray): Hypothesis vector (shape: (N,)).

    Returns:
        int: Always returns 1 (potentially for compatibility with the original code).
    """
    # Exploitation Stage
    svm = SVC(kernel='rbf', random_state=1, C=50, gamma=0.0005)
    svm.fit(data, y)
    ind_S = svm.support_     # Index of support points
    D_S = [data[i] for i in range(len(data)) if i not in ind_S]  # data without support points
    
    S = svm.support_vectors_ # Support points
    N_S = S.shape[0]     # Number of support points
    N_prev = N
    N += allowance
    if N_S <= allowance:
        S = S.tolist()
        R = S
        if len(R) < exploitP*allowance:
            neigh = NearestNeighbors(n_neighbors=2, radius=1)
            neigh.fit(D_S)
            indices_ori = neigh.kneighbors(S, kNS, return_distance=False)
            indices = remove_duplicates_and_sort(indices_ori)
            for i, k in enumerate(indices):
                R.extend(D_S[k])
            indices.sort(reverse=True) # 역순 정렬
            for i, k in enumerate(indices):
                del D_S[k]
    else:
        R = random.sample(S,allowance)
        selected_indices = [S.index(element) for element in R]
        all_indices = set(range(len(S)))
        remaining_indices = all_indices - set(selected_indices)
        for i in remaining_indices:
            D_S.extend(S[i])
               
    # Exploration Stage
    if allowance-len(R) is True:
        Exp = random.sample(D_S, allowance-len(R))
        R.extend(Exp)

    # Update Gram matrix size (if necessary)
    if G.shape[1] < N:
        G.resize((N, N), refcheck=False)

    # Update Gram matrix computed flag
    gramComputed.extend([0]*allowance)

    # Update hypothesis vector (F) (assuming computeGramMatrixCol is implemented)
    F.resize((N,1), refcheck=False)
    non_zero_alpha_indices = [index for index, value in enumerate(alpha) if value != 0]
    for i in range(len(non_zero_alpha_indices)):
        compute_gram_matrix_col(data= data,G = G, g= gamma,idx= non_zero_alpha_indices[i], start_idx= N_prev, gramComputed=gramComputed)  # Call computeGramMatrixCol for updates

    F[N_prev:] = np.dot(G[N_prev:, :N_prev], alpha[:N_prev])
    

    # Update alpha size and fill with zeros
    alpha.resize((N,1), refcheck=False)
    alpha[N_prev:] = 0
    return F, alpha, R 
# if __name__ == '__main__':
    
    # collision check is needed when you change the location of obstacle
print('gram matrix computing..')
G = compute_kernel_gram_matrix(G, data, g)

alpha, F = one_step_weight_kernel_update(alpha, F, data, G, N, g, r, maxUpdate)
print(f"==>> alpha.shape: {alpha.shape}")
print(f"==>> F.shape: {F.shape}")
F, alpha, R = active_learning(data, allowance, kNS, alpha, N, G, gramComputed)
print(f"==>> R.shape: {len(R)}")
print(f"==>> alpha.shape: {alpha.shape}")
print(f"==>> F.shape: {F.shape}")

def collision_check2(theta_data):
    dataset = []
    accumulated_time = 0.0
    
    start = time.time()
    print('collision checking..')
    for l in range(len(theta_data)):
        theta = theta_data[l][0]
        theta2 = theta_data[l][1]
        # theta3 = theta_data[l][2]

        # obstacle(sphere)
        obs_p = np.array([6., 2., 10.])
        obs_r = 3
        obs = colliders.Sphere(obs_p, obs_r)

        # if you want to know distance with link2 and obstacle
        #dist, closest_point_cylinder, closest_point_cylinder2, _ = gjk.gjk(L2, obs)

        # collision check
        # s = np.degrees(theta)
        # d = np.degrees(theta2)
        # f = np.degrees(theta3)
        L1 ,L2 = Arm_with_Mobile(theta, theta2)
        # check_collision_L3 = gjk.gjk_intersection(L3,obs)
      
        check_collision_L2 = gjk.gjk_intersection(L2,obs)
        if check_collision_L2 is True:
            dataset.append([theta,theta2,1])
            # collision.append([s,d,f])
            # result.append(1)
        else:
            check_collision_L1 = gjk.gjk_intersection(L1,obs)
            if check_collision_L1 is True:
                dataset.append([theta,theta2,1])
                # collision.append([s,d,f])
                # result.append(1)
            else:  
                dataset.append([theta,theta2,-1])
                # no_collision.append([s,d,f])
                # result.append(-1)

    end = time.time()
    accumulated_time += end - start
    return dataset

dataset = collision_check2(R)
N = len(dataset)
print(N)
rand_dataset = np.array(dataset)
print(f"==>> dataset.shape: {rand_dataset.shape}")
data = rand_dataset[:, 0:2]
y = rand_dataset[:, [2]] 

alpha, F = one_step_weight_kernel_update(alpha, F, data, G, N, g, r, maxUpdate)

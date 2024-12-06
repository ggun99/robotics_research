import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import itertools

# Dataset Generate
class NLinkArm(object):

    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + self.link_lengths[i - 1] * np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + self.link_lengths[i - 1] * np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


def detect_collision(line_seg, circle):
    a_vec = np.array([line_seg[0][0], line_seg[0][1]])
    b_vec = np.array([line_seg[1][0], line_seg[1][1]])
    c_vec = np.array([circle[0], circle[1]])
    radius = circle[2]
    line_vec = b_vec - a_vec
    line_mag = np.linalg.norm(line_vec)
    circle_vec = c_vec - a_vec
    proj = circle_vec.dot(line_vec / line_mag)
    if proj <= 0:
        closest_point = a_vec
    elif proj >= line_mag:
        closest_point = b_vec
    else:
        closest_point = a_vec + line_vec*proj/line_mag
    if np.linalg.norm(closest_point - c_vec) > radius:
        return False

    return True


def get_occupancy_grid(arm, obstacles):

    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * np.pi / M for i in range(-M // 2, M//2 + 1)]

    dataset = []

    for i in range(M):
        for j in range(M):
            arm.update_joints([theta_list[i], theta_list[j]])
            points = arm.points
            collision_detected = False
            for k in range(len(points) - 1):
                for obstacle in obstacles:
                    line_seg = [points[k], points[k + 1]]
                    collision_detected = detect_collision(line_seg, obstacle)
                    if collision_detected:
                        break
                if collision_detected:
                    break
            grid[i][j] = int(collision_detected)

            if int(collision_detected) == 1:
                collision_stat = 1
            elif int(collision_detected) == 0:
                collision_stat = -1
            dataset.append([theta_list[i], theta_list[j], collision_stat])

    return np.array(grid), dataset


# Simulation parameters
M = 25  # number of sample to divide into and number of grid cell
obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.25], [-1.5, -1.5, 0.25]]  # x y radius
arm = NLinkArm([1, 1], [0, 0])
grid, dataset = get_occupancy_grid(arm, obstacles)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.imshow(grid)

# Fastron
dataset = np.array(dataset)
data = dataset[:, 0:2]
y = dataset[:, [2]]  # preserve shape info


N = data.shape[0]  # number of datapoint = number of row the dataset has
d = data.shape[1]  # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
g = 10  # kernel width
r = 10  # conditional bias
maxUpdate = 200 # max update iteration
maxSupportPoints = 1500  # max support points
G = np.zeros((N, N))  # kernel gram matrix guassian kernel of dataset
alpha = np.zeros((N, 1))  # weight, init at zero
F = np.zeros((N, 1))  # hypothesis
redund = np.zeros((N, 1))
margin = np.zeros((N, 1))

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


def original_kernel_update(alpha, F, data, y, G, N, g, maxUpdate):
    """Brute force update, => unneccessary calculation"""
    for iter in range(maxUpdate):
        print(iter)
        for i in range(N):
            margin = y[i] * hypothesis(data[i], data, alpha, g)
            if margin <= 0:
                alpha[i] += y[i]
                F += y[i] * G[:, [i]]
    return alpha, F

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

def find_min_index(distances):
  """
  N 리스트에서 첫 번째 수 중 가장 작은 수의 인덱스를 찾는 함수입니다.

  Args:
    N: 숫자 리스트입니다.

  Returns:
    가장 작은 수의 인덱스입니다.
  """
  distances = distances.tolist()
  for i in range(len(distances)): 
    for k, row in enumerate(distances):
        min_index = 0
        min_value = distances[i][k]
        if row[k] < min_value:
            min_index = i
            min_value = row[0]

  return min_index

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
    # D_S에서 이 랜덤하게 뽑힌 애들 빼줘야하나  (리셋임?)

    # Update Gram matrix size (if necessary)
    if G.shape[1] < N:
        G.resize((N, N), refcheck=False)

    # Update Gram matrix computed flag
    # print(type(gramComputed))
    # gramComputed = gramComputed.tolist()
    gramComputed.extend([0]*allowance)

    # Update hypothesis vector (F) (assuming computeGramMatrixCol is implemented)
    F.resize((N,1), refcheck=False)
    non_zero_alpha_indices = [index for index, value in enumerate(alpha) if value != 0]
    for i in range(len(non_zero_alpha_indices)):
        compute_gram_matrix_col(data= data,G = G, g= gamma,idx= non_zero_alpha_indices[i], start_idx= N_prev, gramComputed=gramComputed)  # Call computeGramMatrixCol for updates

    F[N_prev:] = np.dot(G[N_prev:, :N_prev], alpha[:N_prev])
    

    # Update alpha size and fill with zeros
    alpha.resize(N, refcheck=False)
    alpha[N_prev:] = 0

print('N',N)
G = compute_kernel_gram_matrix(G, data, g)
# alpha, F = original_kernel_update(alpha, F, data, y, G, N, g, maxUpdate)
alpha, F = one_step_weight_kernel_update(alpha, F, data, G, N, g, r, maxUpdate)
active_learning(data, allowance, kNS, alpha, N, G, gramComputed)

# Test Result
queryP = np.array([1, 1])
collision = eval(queryP, data, alpha, gamma)
print(f"==>> collision: \n{collision}")


def get_occupancy_grid_train():
    MM = 300
    grid = [[0 for _ in range(MM)] for _ in range(MM)]
    theta_list = [2 * i * np.pi / MM for i in range(-MM // 2, MM//2 + 1)]

    for i in range(MM):
        for j in range(MM):
            col = eval(np.array([theta_list[i], theta_list[j]]), data, alpha, g)
            grid[i][j] = int(col)

    return np.array(grid)


grid = get_occupancy_grid_train()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(grid)
plt.show()

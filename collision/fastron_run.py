from collisioncheck2 import ThreeDOF_CollisionCheck as ColCheck
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from fastronWrapper.fastronWrapper import PyFastron

# def data_generation(dataset, alpha = None, gram_computed = None, G = None, F = None):
#     # dataset = Collision.collision_check(theta_data)
#     dataarray = np.array(dataset, dtype=float)
#     data = dataarray[:, 0:3]
#     y = dataarray[:, [3]]
#     N = data.shape[0]  # number of datapoint = number of row the dataset has
#     d = data.shape[1]  # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
#     if alpha is None:
#         alpha = np.zeros((N, 1))  # weight, init at zero
#     if gram_computed is None:
#         gram_computed = np.zeros((N, 1))
#     if G is None:
#         G = np.zeros((N, N))  # kernel gram matrix guassian kernel of dataset
#     if F is None:
#         F = np.zeros((N, 1))  # hypothesis
#     # g = 10  # kernel width
#     # max_updates = 10000  # max update iteration
#     # max_support_points = 800  # max support points
#     # beta = 100
#     # allowance = 1800
#     # kNS = 4
#     # sigma = 2
    
#     return data, y


# def plot(num):
#     if num == 1:
#         range_X = range1_X
#         range_y = range1_y
#     elif num == 2:
#         range_X = range2_X
#         range_y = range2_y
#     else:
#         range_X = range3_X
#         range_y = range3_y
#     C = 1.0  # SVM regularization parameter
#     clf = svm.SVC(kernel='rbf', gamma=5.0, C=C)

#     range_y = range_y.astype(int)
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     range_X_scaled = scaler.fit_transform(range_X)
#     clf.fit(range_X_scaled, range_y)

#     from mlxtend.plotting import plot_decision_regions
#     plot_decision_regions(X=range_X_scaled, y=range_y, clf=clf)

#     plt.scatter(range_X[:, 0], range_X[:, 1], c=range_y, edgecolors='k', marker='o', alpha=0.6)
#     plt.xlabel("scaled joint 1 angle(q1)")
#     plt.ylabel("scaled joint 2 angle(q2)")
#     plt.show()

def get_eval_fastron(fastron,theta_list):
    y_pred = []
    for i in range(len(theta_list)):
                theta = np.array([[theta_list[i][0],theta_list[i][1],theta_list[i][2]]]).astype('float64') #np.array([theta_list[i]]).astype('float64')
                collision_detected = fastron.eval(theta)
                y_pred.append(collision_detected[0])     
    return np.array(y_pred)

#collision check
L1 = 3
L2 = 6
L3 = 6
L_r = 0.1

Collision = ColCheck(L1,L2,L3,L_r)
theta_data, alldata = Collision.theta_list()


for i in range(30):
    

    if i == 0:
        base_x = i
        theta = alldata#theta_data
        dataset, indi ,col = Collision.collision_check(theta,0,2,6,3,base_x)
        # print(f"==>> theta.len: {len(theta)}")
        dataarray = np.array(dataset, dtype=float)
        data = dataarray[:, 0:3]
        y = dataarray[:, [3]]   
    else:
        base_x = i*0.05
        # dataset, indi ,col = Collision.collision_check(data,0,2,6,3,base_x)
        print(f"==>> data1ㄴㄴ.shape: {data1.shape}")
        data_ = np.ndarray.tolist(data1)
        theta = data_
        dataset, _ ,_ = Collision.collision_check(theta,0,2,6,3,base_x)
        # print(f"==>> theta.len: {len(theta)}")
        dataarray = np.array(dataset, dtype=float)
        data = dataarray[:, 0:3]
        y = dataarray[:, [3]]
    
    
    # fastron parameters
    fastron__ = PyFastron(data)
    fastron__.y = y
    fastron__.g = 10
    fastron__.maxUpdates = 5000
    fastron__.maxSupportPoints = 5000
    fastron__.beta = 100
    fastron__.allowance = 500
    fastron__.exploitP = 0.2


    fastron__.updateModel()
    print('here',fastron__.data.shape)
    fastron__.activeLearning()
    print('here',fastron__.data.shape)
    data1 = fastron__.data
    print(f"==>> data1.shape: {data1.shape}")

    y_pred = get_eval_fastron(fastron__, theta)
    # print(f"==>> theta.len: {len(theta)}")
    print(f"==>> y.shape: {y.shape}")
    print(f"==>> y_pred.shape: {y_pred.shape}")
    

    # collision check for all points
    true_alldata, __, __ = Collision.collision_check(alldata,0,2,6,3,base_x)
    pred_alldata = get_eval_fastron(fastron__,alldata)
    # 값이 일치하는 위치 계산
    matches = np.sum(true_alldata == pred_alldata)
    print(f"==>> matches: {matches}")
    # 정확도 계산
    accuracy = matches / len(true_alldata)
    print(f"Accuracy: {accuracy:.2f}")

    # # 값이 일치하는 위치 계산
    # matches = np.sum(y == y_pred)
    # print(f"==>> matches: {matches}")
    # # 정확도 계산
    # accuracy = matches / y.shape[0]
    # print(f"Accuracy: {accuracy:.2f}")

    # X = np.array(data)

    # y_pred = fastron__.eval(data)
    # print(f"==>> y_pred.shape: {y_pred.shape}")
    # # y_pred 
    # z_values = X[:, 2]
    # range1 = np.where(z_values <= 30)
    # range2 = np.where((30 < z_values) & (z_values <= 60))
    # range3 = np.where(60 < z_values)
    # # range1_ = X[range1]
    # # range2_ = X[range2]
    # # range3_ = X[range3]
    # range1_X = X[:,:2][range1]
    # range2_X = X[:,:2][range2]
    # range3_X = X[:,:2][range3]
    # # range1_y = y[:,0][range1]
    # # range2_y = y[:,0][range2]
    # # range3_y = y[:,0][range3]
    # range1_y = y_pred[:,0][range1]
    # range2_y = y_pred[:,0][range2]
    # range3_y = y_pred[:,0][range3]
    
    # plot(3)
    
    # C = 1.0  # SVM regularization parameter
    # clf = svm.SVC(kernel='rbf', gamma=5.0, C=C)

    # range3_y = range3_y.astype(int)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # range3_X_scaled = scaler.fit_transform(range3_X)
    # clf.fit(range3_X_scaled, range3_y)

    # from mlxtend.plotting import plot_decision_regions
    # plot_decision_regions(X=range3_X_scaled, y=range3_y, clf=clf)

    # plt.scatter(range3_X[:, 0], range3_X[:, 1], c=range3_y, edgecolors='k', marker='o', alpha=0.6)
    # plt.xlabel("scaled joint 2 angle(q2)")
    # plt.ylabel("scaled joint 3 angle(q3)")
    # plt.show()


# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import DataLoader, TensorDataset

# np.random.seed(0)
# time_steps = 100
# data = np.sin(np.arange(0, time_steps)) + np.random.normal(0, 0.1, time_steps)

# # 데이터 정규화
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# # 시퀀스 데이터 생성
# def create_sequences(data, time_steps=10):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps])
#     return np.array(X), np.array(y)

# time_steps = 10
# X, y = create_sequences(data_scaled, time_steps)


import pyzed.sl as sl
zed = sl.Camera()
print(zed.get_sdk_version())
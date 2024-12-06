import matplotlib.patches
import numpy as np
import scipy
import scipy.linalg
from matplotlib import pyplot as plt
import matplotlib

l1 = 10
l2 = 10


def Jacobian(th1,th2):
    J = np.array([[-l1*np.sin(th1)-l2*np.sin(th1+th2), -l2*np.sin(th1+th2)],
                 [l1*np.cos(th1)+l2*np.cos(th1+th2), l2*np.cos(th1+th2)]])
    
    return J


th1 = 0.
th2 = 0.

x_1 = l1*np.cos(th1)
y_1 = l1*np.sin(th1)

x_2 = l1*np.cos(th1)+l2*np.cos(th1+th2)
print(f"==>> x_2: {x_2}")
y_2 = l1*np.sin(th1)+l2*np.sin(th1+th2)
print(f"==>> y_2: {y_2}")

J = Jacobian(th1,th2)
J_j = J @ J.T

U, Sigma, V_t = scipy.linalg.svd(J_j)

# Sigma에서 축의 길이 가져오기
sigma_x, sigma_y = Sigma[0], Sigma[1]

# 타원을 그릴 데이터 생성 (기본 단위 원)
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # 단위 원

# 타원으로 변환 (축 길이와 회전 반영)
ellipse = U @ np.diag(np.sqrt(Sigma)) @ circle
x = [0,x_1, x_2]
y = [0,y_1, y_2]
# 그래프 그리기
plt.figure(figsize=(6, 6))
plt.plot(ellipse[0, :]+x_2, ellipse[1, :]+y_2, label='Ellipse', color='blue')  # 타원
plt.plot(x, y)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # x축
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)  # y축
plt.scatter(x_1, y_1, color='red', label='Link1')
plt.scatter(x_2, y_2, color='red', label='Link2')  # 중심점
plt.axis('equal')  # 축 비율 동일
plt.title('Elliptical Representation from SVD')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
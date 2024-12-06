import numpy as np
import matplotlib.pyplot as plt

# 이미지 크기 설정
width, height = 2000,2000

# 랜덤 잔디 색상 생성
grass_color = np.random.rand(height, width, 3) * [0.1, 0.8, 0.1]  # 초록색 계열

# 잔디 텍스처 추가
for _ in range(8000):  # 텍스처의 밀도
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    grass_color[y, x] += np.random.rand(3) * [0.2, 0.5, 0.2]  # 잔디의 랜덤 텍스처 추가

# 이미지 표시
plt.imshow(grass_color)
plt.axis('off')  # 축 숨기기
plt.show()
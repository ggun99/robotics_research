from curses import A_DIM
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode
from sympy import plot

# 1. M, B, K 행렬 정의
M = np.diag([10, 10, 10, 2, 2, 2])  # 질량 행렬
B = np.diag([50, 50, 50, 2, 2, 2])  # 감쇠 행렬
K = np.diag([350, 350, 350, 1, 1, 1])  # 강성 행렬

# 2. 대각선 요소 추출
M_diag = np.diag(M)
B_diag = np.diag(B)
K_diag = np.diag(K)

# 3. 전달 함수 생성 및 Bode Plot 계산
frequencies = np.logspace(-1, 2, 500)  # 주파수 범위 (rad/s)

# Bode plot 데이터 저장용 리스트
magnitude = []
phase = []

# 각 축에 대해 전달 함수 계산
for i in range(len(M_diag)):
    # 강성 값이 0인 경우 작은 값으로 대체
    if K_diag[i] == 0:
        K_diag[i] = 1e-6  # 작은 값 추가

    # 전달 함수 정의 (분자: [1], 분모: [M, B, K])
    num = [1]
    den = [M_diag[i], B_diag[i], K_diag[i]]
    system = TransferFunction(num, den)

    # Bode Plot 계산
    w, mag, ph = bode(system, w=frequencies)
    magnitude.append(mag)
    phase.append(ph)

# 4. Bode Plot 그리기
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8))

# Magnitude Plot
for i, mag in enumerate(magnitude):
    ax_mag.semilogx(w, mag, label=f'Axis {i+1}')
ax_mag.set_title('Bode Plot: Magnitude')
ax_mag.set_ylabel('Magnitude (dB)')
ax_mag.grid(which='both', linestyle='--')
ax_mag.legend()

# Phase Plot
for i, ph in enumerate(phase):
    ax_phase.semilogx(w, ph, label=f'Axis {i+1}')
ax_phase.set_title('Bode Plot: Phase')
ax_phase.set_ylabel('Phase (degrees)')
ax_phase.set_xlabel('Frequency (rad/s)')
ax_phase.grid(which='both', linestyle='--')
ax_phase.legend()

plt.tight_layout()
plt.show()


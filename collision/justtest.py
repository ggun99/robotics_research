import numpy as np

def calculate_accuracy(array1, array2):
    # 두 배열의 길이가 같은지 확인
    if len(array1) != len(array2):
        raise ValueError("두 배열의 길이는 같아야 합니다.")
    
    # numpy 배열로 변환
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # 값이 일치하는 위치 계산
    matches = np.sum(array1 == array2)
    
    # 정확도 계산
    accuracy = matches / len(array1)
    
    return accuracy

# 예제 배열
array1 = [1, 2, 3, 4, 5]
array2 = [1, 2, 0, 4, 5]

# 정확도 계산
accuracy = calculate_accuracy(array1, array2)
print(f"Accuracy: {accuracy:.2f}")
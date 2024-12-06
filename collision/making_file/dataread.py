
def dataset():
    with open("/home/jeongil/collision/making_file/dataset.txt", 'r') as file:
        data = []
        for line in file:
            parts = line.strip().split(',')
            angle1 = float(parts[0])
            angle2 = float(parts[1])
            collision = int(parts[2])
            data.append((angle1, angle2, collision))
        return data



if __name__ == "__main__":
    data = dataset()
    print(data)

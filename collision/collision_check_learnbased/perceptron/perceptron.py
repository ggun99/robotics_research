import numpy as np

def unit_step(x):
    return 1 if x > 0 else 0
    # return np.where(x>0, 1, 0) = if x > 0 return 1 else return 0

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
class Perceptron:
    def __init__(self, learningRate = 0.01, maxIteration = 1000) -> None:
        self.lr = learningRate
        self.iter = maxIteration
        self.activation_func = unit_step
        self.w = None
        self.b = None

    def fit(self, X, y):
        numberOfSample, numberOfFeature = X.shape # assuming row is number of sample, and column is number of feature that an sample have e.g [x1, x2, ..., xn]
        
        self.w = np.zeros(numberOfFeature)
        self.b = 0

        for iter in range(self.iter):
            print(iter)
            for indx, x in enumerate(X):
                output = np.dot(x, self.w) + self.b
                yhat = self.activation_func(output)

                # update
                error = y[indx].item() - yhat
                delw = self.lr * error * x
                delb = self.lr * error
                self.w += delw
                self.b += delb

    def pred(self, x):
        output = np.dot(x, self.w) + self.b
        yhat = self.activation_func(output)
        return yhat

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # EXPERIMENT 1 - AND Logic dataset
    # X = np.array([[1,1],
    #               [1,0],
    #               [0,1],
    #               [0,0]])

    # y = np.array([1,0,0,0]).reshape(4,1)

    # EXPERIMENT 2 - Blob
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


    # Training
    perceptron = Perceptron(learningRate = 0.01, maxIteration=1000)
    perceptron.fit(X_train, y_train)
    print(f"weight = {perceptron.w}, bias = {perceptron.b}")

    # plot separation of boundary line
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-perceptron.w[0] * x0_1 - perceptron.b) / perceptron.w[1]
    x1_2 = (-perceptron.w[0] * x0_2 - perceptron.b) / perceptron.w[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])
    plt.show()
    
    # Query a point
    x = np.array([1,1])
    ypred = perceptron.pred(x)
    print(f"==>> ypred: \n{ypred}")

""" Support Vector machine for classify 2 class of data
https://towardsdatascience.com/understanding-the-hyperplane-of-scikit-learns-svc-model-f8515a109222
"""

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

# we create 20 points seperable in two labels
X = np.array([
    [-1.0, 2.0],
    [-1.5, 2.0],
    [-2.0, 2.0],
    [-1.5, 1.0],
    [-1.5, 1.5],
    [-1.0, 1.5],
    [-1.0, 2.0],
    [2.0, -2.0],
    [2.5, -2.0],
    [3.0, -2.0],
    [2.0, -1.0],
    [2.0, -1.5],
    [2.5, -1.5],
    [3.0, -1.5],
])

y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# we fit the model
clf = svm.SVC(kernel="linear")
clf.fit(X, y)

# we plot the points
plt.scatter(X[:, 0], X[:, 1], c=y, s=20)

# we plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)

# we plot the support vector points
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
# plt.show()

print(clf.coef_)
print(clf.intercept_)

new_point_1 = np.array([[-1.0, 2.5]])
new_point_2 = np.array([[2, -2.5]])
plt.scatter(new_point_1[:, 0], new_point_1[:, 1], c='blue', s=20)
plt.scatter(new_point_2[:, 0], new_point_2[:, 1], c='red', s=20)

plt.show()

print(clf.predict(new_point_1))
print(clf.predict(new_point_2))

# manual calculation (the same as using predict method)
print(np.dot(clf.coef_[0], new_point_1[0]) + clf.intercept_)
print(np.dot(clf.coef_[0], new_point_2[0]) + clf.intercept_)

class Onehot2Int(object):

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.argmax(y_pred, axis=1)
    
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.plotting import plot_decision_regions
from keras.utils import to_categorical

X, y = iris_data()
X = X[:, [2, 3]]

X = standardize(X)

# OneHot encoding
y_onehot = to_categorical(y)

# Create the model
np.random.seed(123)
model = Sequential()
model.add(Dense(8, input_shape=(2,), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='softmax'))

# Configure the model and start training
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])
history = model.fit(X, y_onehot, epochs=10, batch_size=5, verbose=1, validation_split=0.1)

# Wrap keras model
model_no_ohe = Onehot2Int(model)

# Plot decision boundary
plot_decision_regions(X, y, clf=model_no_ohe)
plt.show()
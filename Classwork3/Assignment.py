from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd
'''
# Create dataset A (class 0) and B (class 1)
X1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0, 2.0),
                    cluster_std=0.75,
                    random_state=69)
X2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0, 3.0),
                    cluster_std=0.75,
                    random_state=69)

# Label class A as 0 and class B as 1
y1[:] = 0
y2[:] = 1

# Combine datasets
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Creating a neural network model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=0)

# Making predictions
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob).astype(int).ravel()

# Plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k', cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
'''
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import random

# Generate dataset A (Class 0) and B (Class 1)
X1, y1 = make_blobs(n_samples=100, n_features=2, centers=1,
                    center_box=(-2.0, -2.0), cluster_std=0.75, random_state=42)
X2, y2 = make_blobs(n_samples=100, n_features=2, centers=1,
                    center_box=(2.0, 2.0), cluster_std=0.75, random_state=42)

# Label class A as 0 and class B as 1
y1[:] = 0
y2[:] = 1

# Combine datasets
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# --- Perceptron Section ---
# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perceptron class
class Perceptron:
    def __init__(self, input_size=2, lr=0.01, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.uniform(-1, 1, size=(input_size))  # Random weights
        self.bias = random.uniform(-1, 1)  # Random bias

    def predict(self, X):
        z = np.dot(X, self.w) + self.bias
        return 1 if z > 0 else 0  # Step function

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                self.w += self.lr * error * xi
                self.bias += self.lr * error

# Train the perceptron
perceptron = Perceptron(input_size=2, lr=0.01, epochs=50)
perceptron.fit(X, y)

# Decision boundary function
def decision_function(x1):
    return -(perceptron.w[0] * x1 + perceptron.bias) / perceptron.w[1]

# Generate grid for plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = decision_function(xx)

# --- Plot the Perceptron Decision Boundary ---
plt.figure(figsize=(8, 6))

# Plot dataset
plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Class 1", edgecolor='k')
plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Class 2", edgecolor='k')

# Plot decision boundary line
plt.plot(xx, yy, 'k-', linewidth=2)

# Fill regions
plt.fill_between(xx, yy, y_min, color='lightcoral', alpha=0.3)
plt.fill_between(xx, yy, y_max, color='lightblue', alpha=0.3)

# Plot details
plt.title("Decision Plane")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend(loc='lower right')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

'''

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.model_selection import train_test_split

# # --- Data Generation ---
# # Generate dataset A (Class 0) and B (Class 1)
# X1, y1 = make_blobs(n_samples=100, n_features=2, centers=1,
#                     center_box=(2.0, 2.0), cluster_std=0.75, random_state=42)
# X2, y2 = make_blobs(n_samples=100, n_features=2, centers=1,
#                     center_box=(3.0, 3.0), cluster_std=0.75, random_state=42)

# # Label class A as 0 and class B as 1
# y1[:] = 0
# y2[:] = 1

# # Combine datasets
# X = np.vstack((X1, X2))
# y = np.hstack((y1, y2))

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# # --- Neural Network Model ---
# # Define a simple feedforward neural network
# model = Sequential()
# model.add(Dense(16, input_dim=2, activation='relu'))  # Hidden layer with ReLU activation
# model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation

# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=0)

# # Evaluate the model on test data
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # --- Decision Boundary ---
# # Generate grid points for decision boundary
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))

# # Predict on grid points
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # --- Plot the Decision Boundary ---
# plt.figure(figsize=(8, 6))

# # Plot the filled decision regions
# plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightcoral', 'lightblue'], alpha=0.5)

# # Plot the dataset points
# plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Class 0", edgecolor='k')
# plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Class 1", edgecolor='k')

# # Plot the decision boundary contour
# plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# # Plot details
# plt.title("Neural Network Decision Boundary")
# plt.xlabel("Feature x1")
# plt.ylabel("Feature x2")
# plt.legend(loc='lower right')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# --- Data Generation ---
# Generate dataset A (Class 0) and B (Class 1)
X1, y1 = make_blobs(n_samples=100, n_features=2, centers=1,
                    center_box=(2.0, 2.0), cluster_std=0.75, random_state=42)
X2, y2 = make_blobs(n_samples=100, n_features=2, centers=1,
                    center_box=(3.0, 3.0), cluster_std=0.75, random_state=42)

# Label class A as 0 and class B as 1
y1[:] = 0
y2[:] = 1

# Combine datasets
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# --- Neural Network Model ---
# Define a simple feedforward neural network
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))  # Hidden layer with ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=0)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- Decision Boundary ---
# Generate grid points for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict on grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --- Plot the Decision Boundary ---
plt.figure(figsize=(8, 6))

# Plot the filled decision regions
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightcoral', 'lightblue'], alpha=0.5)

# Plot the dataset points
plt.scatter(X1[:, 0], X1[:, 1], color='red', label="Class 0", edgecolor='k', s=50)
plt.scatter(X2[:, 0], X2[:, 1], color='blue', label="Class 1", edgecolor='k', s=50)

# Plot the decision boundary contour
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Plot details
plt.title("Neural Network Decision Boundary")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend(loc='lower right')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

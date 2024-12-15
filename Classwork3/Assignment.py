from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd

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
plt.contour(xx, yy, Z, levels=[1], colors='black', linewidths=2)

# Plot details
plt.title("Neural Network Decision Boundary")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend(loc='lower right')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()

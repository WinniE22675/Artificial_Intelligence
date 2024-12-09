from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

X1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(-0.25, -0.25),
                    cluster_std=0.25,
                    random_state=69)

X2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(0.75, 0.75),
                    cluster_std=0.25,
                    random_state=69)

# Scale the data using StandardScaler
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
X2_scaled = scaler.fit_transform(X2)

# Define the decision function
def decision_function(x1, x2):
    return x1 + x2 - 0.5

# # Generate a grid of points
x1_range = np.linspace(-3, 3, 500)
x2_range = np.linspace(-3, 3, 500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Evaluate the decision function on the grid
g_values = decision_function(x1_grid, x2_grid)

# Plot the decision boundary and the decision regions
fig = plt.figure()
# plt.figure()
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0 ,np.inf], colors=['red', 'blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black', linewidths=2)

# Plot the dataset
fig.suptitle("Decision Plane")
plt.scatter(X1[:, 0], X1[:, 1], c='purple', linewidths= 1, alpha=0.6, label="Class 1")
plt.scatter(X2[:, 0], X2[:, 1], c='yellow', linewidths= 1, alpha=0.6, label="Class 2")
plt.xlabel('Feature x1', fontsize=10)
plt.ylabel('Feature x2', fontsize=10)
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
fig.savefig('Classwork2/Out1 - Decision Plane.png')


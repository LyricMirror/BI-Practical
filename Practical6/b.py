import numpy as np
import matplotlib.pyplot as plt


def knn_regression(x_train, y_train, new_x, k=3):
    distances = np.abs(x_train - new_x)
    sorted_indices = np.argsort(distances)
    k_nearest_indices = sorted_indices[:k]
    k_nearest_y = y_train[k_nearest_indices]
    predicted_y = np.mean(k_nearest_y)
    return predicted_y


x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])

plt.scatter(x, y, marker='o', color='teal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('KNN Regression')

# Predict for each point in the training data
predicted_y_values = [knn_regression(x, y, xi) for xi in x]
plt.plot(x, predicted_y_values, color='red')

plt.show()

new_x = np.array([6])
predicted_y = knn_regression(x, y, new_x)
print("Predicted y value for new x: ", predicted_y)

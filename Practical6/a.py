import numpy as np
import matplotlib.pyplot as plt

x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])


def lin_reg(x, y):
    n = len(x)
    if len(y) != n:
        raise ValueError("Error")
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    ss_xx = n * np.sum(x * x) - pow(np.sum(x), 2)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return {"slope": slope, "intercept": intercept}


coefficient = lin_reg(x, y)


def predict(x, coefficient):
    return coefficient["slope"] * x + coefficient["intercept"]


plt.scatter(x, y, marker='o', color='teal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression LSM')
plt.plot(x, predict(x, coefficient), color='red')
plt.show()
new_x = np.array([6])
predicted_y = predict(new_x, coefficient)
print("Predicted y values for new x : ", predicted_y)

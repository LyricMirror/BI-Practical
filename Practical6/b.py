import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])


def lin_reg(x, y):
    x = x.reshape(-1, 1)  # Reshape x to a 2D array
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return {"slope": slope, "intercept": intercept}


coefficient = lin_reg(x, y)


def predict(x, coefficient):
    return coefficient["slope"] * x + coefficient["intercept"]


plt.scatter(x, y, marker='o', color='teal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with scikit-learn')
plt.plot(x, predict(x, coefficient), color='red')
plt.show()

new_x = np.array([6])
predicted_y = predict(new_x, coefficient)
print("Predicted y value for new x: ", predicted_y)

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

plt.scatter(x, y, marker='o', color='teal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Naive Bayes Predictions')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted Values')
plt.legend()
plt.show()

print("Gaussian Naive Bayes model accuracy (in %):", metrics.accuracy_score(y_test, y_pred) * 100)

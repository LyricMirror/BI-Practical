from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

gnb = GaussianNB()

x_tr = np.array([4, 5, 2, 6])
y_tr = np.array([3, 7, 2, 72])

x_tr = x_tr.reshape(-1, 1)
y_tr = y_tr.reshape(-1, 1)

gnb.fit(x_tr, y_tr)
y_pred = gnb.predict(x_tr)

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

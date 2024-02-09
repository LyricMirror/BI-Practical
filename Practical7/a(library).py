from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train = np.array([[33.6, 50], [26.6, 31], [23.3, 32], [28.1, 21], [43.1, 33], [25.6, 30], [31, 26], [35.3, 29], [36.5, 53], [37.6, 30]])
y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

X_test = np.array([[45, 30]])
prediction = knn.predict(X_test)

print("Predicted outcome:", prediction[0])

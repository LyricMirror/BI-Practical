import numpy as np


class KNeighborsClassifierCustom:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = np.zeros(len(X), dtype=self.y_train.dtype)
        for i, x in enumerate(X):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_classes = self.y_train[nearest_indices]
            y_pred[i] = np.bincount(nearest_classes).argmax()
        return y_pred


X_train = np.array(
    [[33.6, 50],
     [26.6, 31],
     [23.3, 32],
     [28.1, 21],
     [43.1, 33],
     [25.6, 30],
     [31, 26],
     [35.3, 29],
     [36.5, 53],
     [37.6, 30]])
y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

knn = KNeighborsClassifierCustom(n_neighbors=3)
knn.fit(X_train, y_train)

X_test = np.array([[45, 30]])
prediction = knn.predict(X_test)

print("Predicted outcome:", prediction[0])

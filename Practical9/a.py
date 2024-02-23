import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]
print(iris_df.head())
print(iris_df.describe())
print(iris_df.info())

x = iris_df.iloc[:, [0, 1, 2, 3]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=0, n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

kmeans = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=0, n_clusters=3)
y_means = kmeans.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='red', label='setosa')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='blue', label='versicolour')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=100, c='green', label='virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
plt.legend()
plt.show()

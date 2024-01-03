import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

print(iris_df.head())

# Visualizations
plt.figure(figsize=(15, 10))

# Visualization 1: Scatterplot
plt.subplot(2, 4, 1)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=iris_df)
plt.title('sepal length (cm) vs sepal width (cm)')

# Visualization 2: Histogram
plt.subplot(2, 4, 2)
sns.histplot(data=iris_df, x='petal length (cm)', hue='species', kde=True)
plt.title('petal length (cm) Distribution')

# Visualization 3: KDE Plot
plt.subplot(2, 4, 3)
sns.kdeplot(data=iris_df, x='petal width (cm)', hue='species')
plt.title('petal width (cm) KDE')

# Visualization 4: Boxplot
plt.subplot(2, 4, 4)
sns.boxplot(x='species', y='sepal width (cm)', data=iris_df)
plt.title('sepal width (cm) Distribution')

# Visualization 5: Violin Plot
plt.subplot(2, 4, 5)
sns.violinplot(x='species', y='sepal length (cm)', data=iris_df)
plt.title('sepal length (cm) Distribution')

# Visualization 6: Jointplot
plt.subplot(2, 4, 6)
sns.jointplot(x='petal width (cm)', y='petal length (cm)', data=iris_df, kind='scatter')
plt.title('petal width (cm) vs petal length (cm)')

# Visualization 7: PairGrid
plt.subplot(2, 4, 7)
g = sns.PairGrid(iris_df, hue='species', palette='husl')
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot)
g.add_legend()
plt.title('PairGrid')

plt.tight_layout()
plt.show()

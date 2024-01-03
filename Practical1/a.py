import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r'C:\Users\Lenovo\PycharmProjects\BIPractical\DataFile\weather_game.csv'
data = pd.read_csv(file_path)


print("First 5 rows of the dataset:")
print(data.head())


print("\nSummary Statistics:")
print(data.describe())


print("\nData information:")
print(data.info())


plt.figure(figsize=(8, 6))
sns.countplot(x='Outlook', data=data)
plt.title('Count of Outlook')
plt.xlabel('Outlook')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='Wind', data=data)
plt.title('Count of Wind')
plt.xlabel('Wind')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(data['Temperature'], bins=10, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(data['Humidity'], bins=10, kde=True)
plt.title('Humidity Distribution')
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.show()

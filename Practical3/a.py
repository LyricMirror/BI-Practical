import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file_path = r'C:\Users\Lenovo\PycharmProjects\BIPractical\DataFile\weather_game.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

data['Temperature'].fillna(data['Temperature'].mean(), inplace=True)
data['Outlook'] = data['Outlook'].astype('category').cat.codes
data['Wind'] = data['Wind'].map({'weak': 0, 'strong': 1})
data['Play'] = data['Play'].map({'no': 0, 'yes': 1})

cleaned_file_path = r'C:\Users\Lenovo\PycharmProjects\BIPractical\DataFile\cleaned_weather_data.csv'
data.to_csv(cleaned_file_path, index=False)

numerical_columns = ['Outlook', 'Temperature', 'Wind', 'Play']
min_max_scaler = MinMaxScaler()
data[numerical_columns] = min_max_scaler.fit_transform(data[numerical_columns])
z_score_scaler = StandardScaler()
data[numerical_columns] = z_score_scaler.fit_transform(data[numerical_columns])

print('\n\nAfter Cleaning and Normalization\n\n')
print('First 5 Rows : ')
print(data.head())

print('\nSummary statistic of the dataset : ')
print(data.describe())

missing_values = data.isnull().sum()
print('\nMissing values After Cleaning and Normalization : ')
print(missing_values)

# b.py part
print("\n\nAfter Cleaning\n\n")
print("First 5 rows of the dataset:")
print(data.head())

numerical_columns = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play']

print("\nSummary statistics of the dataset after cleaning:")
print(data[numerical_columns].describe())

missing_values = data[numerical_columns].isnull().sum()
print("\nMissing values in the numerical columns after cleaning:")
print(missing_values)

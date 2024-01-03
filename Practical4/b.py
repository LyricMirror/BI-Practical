import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset from the URL
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRQZeodQ2VjJUROTmPZ_IbQINhQ6xklABgmwk4pEH69gtVb6MAfsuIlH9bjvxL6WNaB1QX5ZhxYTsSJ/pub?gid=1565081751&single=true&output=csv"
sales_data = pd.read_csv(url)
print(sales_data.head())
# Adjusting figure size for subplots
plt.figure(figsize=(18, 12))

# Bar Chart of Country-wise Sales
plt.subplot(3, 3, 1)
top_countries = sales_data['Country'].value_counts().head(10)
top_countries.plot(kind='bar')
plt.title('Top 10 Countries by Sales')

# Histogram of Quantity Purchased
plt.subplot(3, 3, 2)
plt.hist(sales_data['Quantity'], bins=30)
plt.title('Distribution of Quantity Purchased')

# Line Plot of Sales Over Time
plt.subplot(3, 3, 3)
sales_data['InvoiceDate'] = pd.to_datetime(sales_data['InvoiceDate'])
monthly_sales = sales_data.groupby(sales_data['InvoiceDate'].dt.to_period('M')).size()
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')

# Sales Distribution by Product Description
plt.subplot(3, 3, 4)
top_products = sales_data['Description'].value_counts().head(10)
top_products.plot(kind='bar')
plt.title('Top 10 Products by Sales')

# Sales Count by Customer ID
plt.subplot(3, 3, 5)
sales_data['CustomerID'].value_counts().plot(kind='hist', bins=30)
plt.title('Sales Count by Customer ID')

# Sales Variation by Day of the Week
plt.subplot(3, 3, 6)
sales_data['DayOfWeek'] = sales_data['InvoiceDate'].dt.day_name()
sales_by_day = sales_data['DayOfWeek'].value_counts()
sales_by_day.plot(kind='bar')
plt.title('Sales Variation by Day of the Week')

# Monthly Sales Trend
plt.subplot(3, 3, 7)
monthly_sales = sales_data.resample('M', on='InvoiceDate').InvoiceNo.count()
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')

# Top Countries by Sales
plt.subplot(3, 3, 8)
top_countries = sales_data.groupby('Country').size().sort_values(ascending=False).head(10)
top_countries.plot(kind='bar')
plt.title('Top 10 Countries by Sales')

# Sales by Hour of the Day
plt.subplot(3, 3, 9)
sales_data['Hour'] = sales_data['InvoiceDate'].dt.hour
sales_by_hour = sales_data.groupby('Hour').size()
sales_by_hour.plot(kind='bar')
plt.title('Sales by Hour of the Day')

# Adjust layout
plt.tight_layout()
plt.show()

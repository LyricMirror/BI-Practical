import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'http://quotes.toscrape.com/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')


quotes = []
authors = []
for quote in soup.find_all('span', class_='text'):
    quotes.append(quote.get_text())
for author in soup.find_all('small', class_='author'):
    authors.append(author.get_text())


data_df = pd.DataFrame({'Quote': quotes, 'Author': authors})
author_counts = data_df['Author'].value_counts()


plt.figure(figsize=(10, 6))
sns.barplot(x=author_counts.index, y=author_counts.values, hue=author_counts.index, palette="viridis", legend=False)
plt.title('Number of Quotes by Author')
plt.xlabel('Author')
plt.ylabel('Number of Quotes')
plt.xticks(rotation=45)
plt.show()

data_df.to_csv(r'C:\Users\Lenovo\PycharmProjects\BIPractical\DataFile\scraped_quotes.csv', index=False)
print("Scraped data has been saved to 'scraped_quotes.csv'.")

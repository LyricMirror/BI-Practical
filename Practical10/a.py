import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = [
    ['bread', 'cheese', 'egg', 'juice'],
    ['bread', 'cheese', 'juice'],
    ['milk', 'bread', 'yogurt'],
    ['bread', 'milk', 'juice'],
    ['cheese', 'milk', 'juice'],
]

te = TransactionEncoder()
te_ary = te.fit_transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets_1 = apriori(df, min_support=0.5, use_colnames=True)
print("Frequent One-Item Sets : ")
print(frequent_itemsets_1)
frequent_itemsets_1 = frequent_itemsets_1[frequent_itemsets_1['support'] >= 0.5]

frequent_itemsets_2 = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets_2['length'] = frequent_itemsets_2['itemsets'].apply(lambda x: len(x))
frequent_itemsets_2 = frequent_itemsets_2[frequent_itemsets_2['length'] == 2]
print("\nFrequent Two-Item Sets : ")
print(frequent_itemsets_2)
frequent_itemsets_2 = frequent_itemsets_2[frequent_itemsets_2['support'] >= 0.5]

conditional_probabilities = {}
final_pairs = []

for index, row in frequent_itemsets_2.iterrows():
    item1, item2 = list(row['itemsets'])
    support_both = row['support']

    support_item1 = frequent_itemsets_1[frequent_itemsets_1['itemsets'].apply(lambda x: item1 in x)]['support'].values[0]
    support_item2 = frequent_itemsets_1[frequent_itemsets_1['itemsets'].apply(lambda x: item2 in x)]['support'].values[0]

    conditional_prob_item1_given_item2 = support_both / support_item1
    conditional_prob_item2_given_item1 = support_both / support_item2

    if conditional_prob_item1_given_item2 >= 0.5 and conditional_prob_item2_given_item1 >= 0.5:
        final_pairs.append((item1, item2, conditional_prob_item1_given_item2, conditional_prob_item2_given_item1))

final_pairs_df = pd.DataFrame(final_pairs, columns=['items1', 'items2', 'Confidence_Item1_to_Item2', 'Confidence_Item2_to_Item1'])
print("\nFinal Pairs : ")
print(final_pairs_df)

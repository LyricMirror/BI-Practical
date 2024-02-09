import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


data = {
    'colour': ['red', 'red', 'red', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'red', 'red'],
    'type': ['sports', 'sports', 'sports', 'sports', 'sports', 'SUV', 'SUV', 'SUV', 'SUV', 'sports'],
    'origin': ['domestic', 'domestic', 'domestic', 'domestic', 'imported', 'imported', 'imported', 'domestic', 'imported', 'imported'],
    'stolen': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes']
}
label_encoder = LabelEncoder()
df = pd.DataFrame(data)
df['colour'] = label_encoder.fit_transform(df['colour'])
df['type'] = label_encoder.fit_transform(df['type'])
df['origin'] = label_encoder.fit_transform(df['origin'])
df['stolen'] = label_encoder.fit_transform(df['stolen'])

X = df[['colour', 'type', 'origin']]
y = df['stolen']

model = CategoricalNB()
model.fit(X, y)

new_data = {'colour': ['red'], 'type': ['SUV'], 'origin': ['domestic']}
new_df = pd.DataFrame(new_data)
new_df['colour'] = label_encoder.transform(new_df['colour'])
new_df['type'] = label_encoder.transform(new_df['type'])
new_df['origin'] = label_encoder.transform(new_df['origin'])

prediction = model.predict(new_df)

predicted_class = label_encoder.inverse_transform(prediction)
print("Predicted class:", predicted_class)

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

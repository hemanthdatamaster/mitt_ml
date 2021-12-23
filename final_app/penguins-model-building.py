import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

data = penguins.copy()


target = 'species'
encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data,dummy], axis=1)
    del data[col]
    

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]


# Separating X and y
X = data.drop('species', axis=1)
Y = data['species']

# Build random forest model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X, Y)

# Saving the model
import pickle
pickle.dump(model, open('penguins_clf.pkl', 'wb'))

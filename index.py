import pandas as pd
import numpy as np
# ML Packages

#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Load our data
df = pd.read_csv('Popular_Baby_Names.csv')
print(df.size)
print(df.columns)
print(df.shape)

print(df['Name'].unique())

print(df.groupby('Name')['year'].size())


# print the plot about the data
# print(df.groupby('Name')['year'].size().plot(kind='bar',figsize=(20,10)))



# Features
Xfeatures = df['Name']
ylabels= df['year']

#Xfeatures.values.reshape(-1, 1)
# Vectorize Features
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

print(cv.get_feature_names())

x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.33, random_state=42)
## Build Model

nv = MultinomialNB()
nv.fit(x_train,y_train)


# Accuracy of Our Model
print(nv.score(x_test,y_test))

sample1 = ["SOFIA"]

vect1 = cv.transform(sample1)
print(nv.predict(vect1))
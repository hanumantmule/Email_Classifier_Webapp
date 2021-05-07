import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# reading email datset which is stored on google drive
data = pd.read_csv('./spam_ham_dataset.csv')

# dropping the ID column as it is of no use in analysis
data.drop('Unnamed: 0', axis=1, inplace=True)

vectorizer = CountVectorizer()
spam_ham = vectorizer.fit_transform(data['text'])

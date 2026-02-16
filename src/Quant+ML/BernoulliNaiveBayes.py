import  numpy as np
import  pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


df= pd.read_csv("spam_ham_dataset.csv")
# print(df.shape)
print(df.columns)
df= df.drop(['Unnamed: 0'], axis=1)
print(df)

X=df["text"].values
y=df["label_num"].values

cv = CountVectorizer()
X= cv.fit_transform(X)

X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.3, random_state=42)
bnb= BernoulliNB(binarize=0.0)
model = bnb.fit(X_train, y_train)

y_pred=  bnb.predict(X_test)

print(classification_report(y_test, y_pred))


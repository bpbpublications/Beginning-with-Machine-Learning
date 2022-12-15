import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing


data = pd.read_csv("titanic_train.csv")
data.head()
data=data.drop(['Name'], axis=1)
Label_encoder=preprocessing.LabelEncoder()
Sex_encoded=Label_encoder.fit_transform(data["Sex"])
data=data.drop(['Sex'], axis=1)
X_train, X_test, y_train, y_test = train_test_split ( data.drop('Survived',axis = 1), data['Survived'], test_size = 0.3, random_state = 101)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

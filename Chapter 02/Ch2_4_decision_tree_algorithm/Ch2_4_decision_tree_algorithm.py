import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv("kyphosis.csv")
data.head()

X = data.drop("Kyphosis", axis=1)
y = data["Kyphosis"]
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size =0.3, random_state=101)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


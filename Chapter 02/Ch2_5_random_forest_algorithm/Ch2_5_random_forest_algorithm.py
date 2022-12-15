import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv("kyphosis.csv")
data.head()

X = data.drop("Kyphosis", axis=1)
y = data["Kyphosis"]
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size =0.3, random_state=101)

model = RandomForestClassifier(n_estimators = 250)
model.fit(X_train, y_train)

RFC_predictions = model.predict(X_test)
print(confusion_matrix(y_test, RFC_predictions))
print(classification_report(y_test, RFC_predictions))
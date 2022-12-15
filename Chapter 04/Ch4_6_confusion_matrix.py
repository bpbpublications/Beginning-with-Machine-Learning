import pandas as pd
import numpy as np
from sklearn import dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svc = SVC(kernel='linear', C=10.0, random_state=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(conf_matrix)
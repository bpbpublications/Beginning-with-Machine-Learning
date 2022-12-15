import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
%matplotlib inline

data = pd.read_csv("Classified Data",index_col=0)
data.head()

SS = StandardScaler()
SS.fit(df.drop('TARGET CLASS', axis = 1))
SS_features = SS.transform(df.drop('TARGET CLASS', axis = 1))
df_features = pd.DataFrame(SS_features, columns = df.columns[:-1])
df_features.head()

X_train, X_test, y_train, y_test = train_test_split(SS_features,df['TARGET CLASS'], test_size = 0.30)

model = KneighborsClassifier(n_neighbors = 1)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(accuracy_score(y_test, pred))
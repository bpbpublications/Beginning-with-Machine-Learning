import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv("USA_Housing.csv")
data.head()

X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('MSE:', metrics.mean_squared_error(y_test, predictions))
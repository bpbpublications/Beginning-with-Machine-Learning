import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

A = load_breast_cancer()
X = A.data
Y = A.target
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size =0.30, random_state = 101)

model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions)) 

from sklearn.model_selection import GridSearchCV
param_grid = { 'C': [0.1, 1, 10, 100],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'gamma':['scale', 'auto'],'kernel': ['linear'] }

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, n_jobs= -1) 

grid.fit(X_train, y_train)
print(grid.best_params_)
grid_predictions = grid.predict(X_test)

print(classification_report(y_test, grid_predictions))

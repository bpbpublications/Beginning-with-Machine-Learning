from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn import datasets

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
lr = linear_model.LogisticRegression()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr.fit(X_train_fold, y_train_fold)
    lst_accu_stratified.append(lr.score(X_test_fold, y_test_fold))

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test_fold)
cm = confusion_matrix(y_test_fold, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train_
fold, y = y_train_fold, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
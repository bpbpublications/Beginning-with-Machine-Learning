from sklearn.metrics import mean_absolute_error

y_true = [4, -0.7, 3, 6]
y_pred = [3.5, 0.1, 1.8, 6]
MAE = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:")
print(MAE)
from sklearn.metrics import mean_squared_error
import math

y_true = [4, -0.7, 3, 6]
y_pred = [3.5, 0.1, 1.8, 6]
MSE = mean_squared_error(y_true, y_pred)
print("Mean Square Error: ")
print(MSE) 
RMSE = math.sqrt(MSE)
print ('Root Mean Square Error :')
print(RMSE)
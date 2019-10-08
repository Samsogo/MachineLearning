from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
data = boston.data
target = boston.target

x = data[:,5]
y = target

x = x[y<50]
y = y[y<50]



x_train,x_test,y_train,y_test = train_test_split(x,y)

x_train_mean = np.mean(x_train)
y_train_mean = np.mean(y_train)

num = (x_train - x_train_mean).dot(y_train - y_train_mean)
fm = (x_train - x_train_mean).dot(x_train - x_train_mean)

a = num / fm
b = y_train_mean - a*x_train_mean


y_predict = a * x_test + b


mse = mean_squared_error(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

print(mse)
print(mae)
print(r2)
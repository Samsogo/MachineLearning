'''
衡量简单线性回归算法的指标：
    （1）均方误差 MSE（Mean Squared Error）
    （2）均方根误差 RMSE（Root Mean Squared Error）
    （3）平均绝对误差 MAE（Mean Absolute Error）
'''

'''
我们使用sklearn自带的波士顿房产数据进行测试
'''

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()
data = boston.data

x = data[:,5]  #取的第五列是每个房子的房间数量特征
y = boston.target

#去除y大于等于50的数据，也就是不准确数据
x = x[y<50]
y = y[y<50]


x_train,x_test,y_train,y_test = train_test_split(x,y)

x_train_mean = np.mean(x_train)
y_train_mean = np.mean(y_train)

num = (x_train - x_train_mean).dot(y_train - y_train_mean)
fm = (x_train - x_train_mean).dot(x_train - x_train_mean)

a = num / fm
b = y_train_mean - a * x_train_mean

print("a:",a)
print("b:",b)

Y = a * x_train + b
plt.scatter(x_train,y_train,color='r')
plt.plot(x_train,Y,color='b')
plt.show()

'''
    测试算法好坏
'''

y_predict = a * x_test + b

mse = np.sum((y_predict - y_test) ** 2) / len(y_test)
rmse = np.sqrt(mse)
mae = np.sum(np.absolute(y_predict - y_test)) / len(y_test)

print("MSE:",mse)
print("RMSE:",rmse)
print("MAE",mae)

'''
    线性回归的重要的测试模型好坏的指标
    R^2 （R Squared）= 1 - MSE/Var(y_test)
    R^2越大越好
'''

r2 = 1 - mse/np.var(y_test)
print("R^2:",r2)

r = r2_score(y_test,y_predict)
print(r)







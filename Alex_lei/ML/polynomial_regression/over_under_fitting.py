'''

主要说的是机器学习中一个很常见的问题
    过拟合：过多的表达了数据之间的关系
    欠拟合：较少的表达了数据之间的关系

案例：使用之前的多项式回归的随机的数据集

'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

m = 100
x = np.random.uniform(-3,3,size=m)
y = 1.5 * x ** 2 + 2 * x + 2 + np.random.normal(0,1,size=m)
X = x.reshape(-1,1)


def PolynomialRegression(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('stand',StandardScaler()),
        ('lin_reg',LinearRegression())
    ])


#首先说下欠拟合
lin_reg = LinearRegression()
lin_reg.fit(X,y)
y_predict = lin_reg.predict(X)
mse = mean_squared_error(y,y_predict) #该值较大，从图也可以看出拟合程度并不是很好
print(mse)
# plt.scatter(x,y)
# plt.plot(np.sort(x),y_predict[np.argsort(x)],c='r')
# plt.title('under fitting')
# plt.show()


#再说下过拟合

poly_reg = PolynomialRegression(degree=200)
poly_reg.fit(X,y)
y_predict2 = poly_reg.predict(X)
mse2 = mean_squared_error(y,y_predict2)
print(mse2)
#当degree的值越来越大的时候，该值越来越小，但是这样的现象看起来训练集拟合的很好，
# 但是使用测试集进行测试的时候会出现问题，因为这样拟合的结果不能代表我们整个数据集的走向趋势
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],c='r')
plt.title('over fitting')
plt.show()


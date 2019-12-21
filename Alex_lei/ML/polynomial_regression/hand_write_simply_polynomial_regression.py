import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


m = 100
n = 1
x = np.random.uniform(-4,4,size=m)
X = x.reshape(-1,1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0,1,size=m)

#使用线性回归
lin_reg = LinearRegression()
lin_reg.fit(X,y)
y_predict = lin_reg.predict(X)

#加一列，使用的是多项式回归，但是算法还是使用的线性回归
x1 = X ** 2
X2 = np.hstack([X,x1])
lin_reg2 = LinearRegression()
lin_reg2.fit(X2,y)
y_predict2 = lin_reg2.predict(X2)

print(lin_reg2.coef_)
print(lin_reg2.intercept_)

plt.scatter(x,y)
plt.plot(x,y_predict,c='r')
plt.plot(np.sort(x),y_predict2[np.argsort(x)],c='g')
plt.show()

#从结果可以看出绿色的拟合效果更好些
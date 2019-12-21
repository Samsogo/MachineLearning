from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(666)
m = 100
x = np.random.uniform(-4,4,size=m)
y = 0.5 * x ** 2 + 2 * x + 1 + np.random.normal(0,1,size=m)
X = x.reshape(-1,1)


poly_reg = PolynomialFeatures(degree=2)
poly_reg.fit(X,y)
X2 = poly_reg.transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X2,y)
y_predict = lin_reg.predict(X2)
print(lin_reg.intercept_) #截距
print(lin_reg.coef_) #系数
print(lin_reg.score(X2,y))
score = mean_squared_error(y,y_predict)
score2 = np.sum((y-y_predict)**2) / m
print(score)
print(score2)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],c='r')
plt.show()
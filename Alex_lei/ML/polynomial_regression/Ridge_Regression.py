
'''

主要说的是岭回归，是模型正则化的一种方法，让我们的模型拟合效果更好

'''

import  numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

x = np.random.uniform(-3,3,size=100)
X = x.reshape(-1,1)
y = 0.5 * x + 3 + np.random.normal(0,1,size=100)
Y = y
x_train, x_test, y_train, y_test = train_test_split(X,Y)

def plot_mode(model,s):
    X_plot = np.linspace(-3,3,100).reshape(100,1)
    y_plot = model.predict(X_plot)
    plt.scatter(x,y)
    plt.plot(X_plot[:,0],y_plot,c='r')
    plt.axis([-3,3,0,6])
    plt.title(s)
    plt.show()

def pipeline(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('stand',StandardScaler()),
        ('lin_reg',LinearRegression())
    ])

poly_reg = pipeline(20)
poly_reg.fit(x_train,y_train)
y_predict = poly_reg.predict(x_test)
score1 = mean_squared_error(y_test,y_predict)
print(score1)
plot_mode(poly_reg,'poly_reg')

def pipeline2(degree,alpha):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('stand',StandardScaler()),
        ('ridge_reg',Ridge(alpha=alpha))
    ])

ridge_reg = pipeline2(20,1)
ridge_reg.fit(x_train,y_train)
y_predict2 = ridge_reg.predict(x_test)
score2 = mean_squared_error(y_test,y_predict2)
print(score2)

plot_mode(ridge_reg,'ridge_reg')
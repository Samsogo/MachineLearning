import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


m = 100
x = np.random.uniform(-3,3,size=m)
X = x.reshape(-1,1)
y = 0.5 * x + 3 + np.random.normal(0,1,size=m)


x_train, x_test, y_train, y_test = train_test_split(X,y)

def Poly_pipeline(degree):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("stand",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])

def Lasso_regression(degree,alpha):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("stand",StandardScaler()),
        ("lasso",Lasso(alpha=alpha))
    ])

def plot_mode(model,s):
    X_plot = np.linspace(-3,3,100).reshape(100,1)
    y_plot = model.predict(X_plot)
    plt.scatter(X,y)
    plt.plot(X_plot[:,0],y_plot,c='r')
    plt.axis([-3,3,0,6])
    plt.title(s)
    plt.show()

if __name__ == '__main__':
    poly = Poly_pipeline(20)
    poly.fit(x_train,y_train)
    y_predict = poly.predict(x_test)
    poly_score = mean_squared_error(y_test,y_predict)
    print(poly_score)
    lasso = Lasso_regression(20,0.0001)
    lasso.fit(x_train,y_train)
    lasso_y_predict = lasso.predict(x_test)
    lasso_score = mean_squared_error(y_test,lasso_y_predict)
    print(lasso_score)
    plot_mode(poly,"线性回归")
    plot_mode(lasso,"lasso回归")
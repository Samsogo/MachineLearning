# -*- coding: utf-8 -*-

'''

主要是测试模型的泛化能力
当训练集的拟合程度很好，并且得到的模型对测试集也拟合的很好，这样的模型泛化能力是很强的
当训练集的拟合程度很好，但是得到的模型对测试集拟合的很差，这样的模型泛化能力是很弱的

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

m = 100
x = np.random.uniform(-5,5,size=m)
y = 0.5 * x ** 2 + 2 * x + 5 + np.random.normal(0,1,size=m)

X = x.reshape(-1,1)

def polynomialRegress(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('stand',StandardScaler()),
        ('lin_reg',LinearRegression())
    ])

def testModelGeneralizationCapability(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X,y)
    de = []
    sc_test = []
    sc_train = []
    for i in range(0,5):
        degree = i
        de.append(degree)
        poly_reg = polynomialRegress(degree)
        poly_reg.fit(x_train,y_train)
        y_predict_test = poly_reg.predict(x_test)
        m1 = mean_squared_error(y_test,y_predict_test)
        score_test = (y_test,y_predict_test)
        y_predict_train = poly_reg.predict(x_train)
        m2 = mean_squared_error(y_train,y_predict_train)
        score_train = r2_score(y_train,y_predict_train)
        sc_test.append(1 - m1)
        sc_train.append(1 - m2)

    return de,sc_train,sc_test

if __name__ == '__main__':
    de,score_train,score_test = testModelGeneralizationCapability(X,y)
    score_train = (score_train - np.min(score_train)) / (np.max(score_train) - np.min(score_train))
    score_test = (score_test - np.min(score_test)) / (np.max(score_test) - np.min(score_test))
    plt.plot(de,score_train,c='g',label='train')
    plt.plot(de,score_test,c='r',label='test')
    print(score_train)
    print(score_test)
    plt.legend()
    plt.title('Find the best place for generalization')
    plt.show()
    #通过这个图像可以看出，横轴是模型复杂度，纵轴表示模型准确率，
    #训练集的线会一直上升，而测试集在一个点之前会一直上涨，到达这个点之后就会下降，所以该点是整个模型泛化能力最强的那个点



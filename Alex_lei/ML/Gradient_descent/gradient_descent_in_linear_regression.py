'''

    梯度下降法在简单线性回归中的应用

'''

import numpy as np

x = np.random.random(size=100)
y = 3 * x + 2.5 + np.random.normal(size=100)

def J(theta,X_b,y):
    return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)

def DJ(theta,X_b,y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)

def Gradient_descent(theta,eta,X_b,y,n_iters = 1000):

    while True:
        last_theta = theta
        theta =  theta - eta * DJ(theta,X_b,y)
        if np.abs(J(theta,X_b,y) - J(last_theta,X_b,y)) < 1e-8:
            break
    return theta

if __name__ == '__main__':
    x = x.reshape(-1,1)
    X_b = np.hstack([np.ones((len(x),1)),x])
    theta = np.zeros(X_b.shape[1])
    res_theta = Gradient_descent(theta, 0.01, X_b, y)
    print(res_theta[0]) #截距
    print(res_theta[1]) #斜率

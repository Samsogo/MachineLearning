'''

    验证我们计算的回归参数的值是否正确
    核心知识点就是一个点的导数和这个点的相邻的两个点的y差值除以x差值，导数的定义也就是

'''


import numpy as np

np.random.seed(666)
x = np.random.random(size=(1000, 10))
true_theta = np.arange(1,12,dtype=float)
X_b = np.hstack([np.ones((len(x),1)),x])
y = X_b.dot(true_theta) + np.random.normal(size=1000)

def J(X_b,theta,y):
    return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)

def DJ_math(X_b,theta,y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. /len(X_b)

def DJ_debug(X_b,theta,y,epsilon=0.000000001):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2 -= epsilon
        res[i] = (J(X_b,theta_1,y) - J(X_b,theta_2 ,y)) / (2 * epsilon)
    return res

def gd(dj,X_b, theta, y):
    while True:
        last_theta = theta
        theta = theta - 0.01 * dj(X_b,theta,y)
        if np.abs(J(X_b,theta,y) - J(X_b,last_theta,y)) < 1e-8:
            break
    return theta


X_b = np.hstack([np.ones((len(x),1)),x])
theta = np.zeros(X_b.shape[1])
theta = gd(DJ_math,X_b,theta,y)
theta_bak = gd(DJ_debug,X_b,theta,y)
print(theta)
print(theta_bak)
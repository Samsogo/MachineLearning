'''

    随机梯度下降法在简单线性回归中的应用

'''

import numpy as np

x = np.random.random(size=100000)
y = 3 * x + 2.5 + np.random.normal(size=100000)

def J(theta,X_b,y):
    return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)

def DJ(theta,X_b_i,y_i):
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2 / len(X_b_i)


#使用了1/3的样本进行计算theta
def Gradient_descent(theta,X_b,y,n_iters):

    def getEta(i):
        return 5 / (i+50)
    for i in range(0,n_iters):
       index = np.random.randint(0,len(X_b))
       theta = theta - getEta(i) * DJ(theta,X_b[index],y[index])
    return theta

#使用1/3进行计算显然是不合理的，所以采用下面的方法
def Gradient_descent_new(theta,X_b,y,n):
    def getEta(i):
        return 5 / (i+50)
    m = len(X_b)
    for i in range(0,n):
        randomIndex = np.random.permutation(m)
        X_b_new = X_b[randomIndex]
        y_new = y[randomIndex]
        for j in range(0,m):
            theta = theta - getEta(i*m+j) * DJ(theta,X_b_new[j],y_new[j])
    return theta

if __name__ == '__main__':
    x = x.reshape(-1,1)
    X_b = np.hstack([np.ones((len(x),1)),x])
    theta = np.zeros(X_b.shape[1])
    res_theta = Gradient_descent(theta, X_b, y,len(X_b) // 3)
    print(res_theta[0]) #截距
    print(res_theta[1]) #斜率

    theta = np.zeros(X_b.shape[1])
    res_theta1 = Gradient_descent_new(theta,X_b,y,5)
    print(res_theta1)

'''

数据如何降维的：
    我们求出前k个主成分，
    样本中的一个样本映射到k个主成分之后，得到的就是降维后的样本

'''

import numpy as np
import matplotlib.pyplot as plt

m = 1000
n = 2
x = np.empty((m,n))
x[:,0] = np.random.uniform(0,100,size=m)
x[:,1] = 0.75 * x[:,0] + 3 + np.random.normal(0,10,size=m)

def f(x,w):
    return np.sum((x.dot(w)) ** 2) / m

def df(x,w):
    return x.T.dot(x.dot(w)) * 2 / m

def demean(x):
    return x - np.mean(x,axis=0)

def direction(w):
    return w / np.linalg.norm(w)

def first_compents(x,w,eta=0.001,n_iters=1e4,epsilon=1e-8):
    w = direction(w)
    while True:
        last_w = w
        w = w + eta * df(x,w)
        w = direction(w)
        if abs(f(x,last_w) - f(x,w)) < epsilon:
            break

    return w

def transform(x,w):
    return x.dot(w.T)

if __name__ == '__main__':
    w = np.random.random(size=n)
    x1 = demean(x)
    res_w = first_compents(x1,w)
    res_w = res_w.reshape(1,-1)
    x_new = transform(x,res_w)  #降维后的样本
    x_old = x_new.dot(res_w) #返回降维之前的样本，由于pca会有损失，所以不能和原来的样本完全一样
    plt.scatter(x_old[:,0],x_old[:,1],c='r')
    plt.scatter(x[:,0],x[:,1])
    plt.show()
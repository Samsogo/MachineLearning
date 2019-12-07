'''

pca 是一个非监督学习算法
主要用于降维和去噪
方法主要是将特征映射到一个轴上，使得样本的间距最大，可以使用方差来描述这个概念

'''

import numpy as np
import matplotlib.pyplot as plt

m = 1000
n = 2
x = np.empty([m,n])
x[:,0] = np.random.uniform(0,100,size=m)
x[:,1] = 0.75 * x[:,0]+3 + np.random.normal(size=m)

def f(x,w):
    return np.sum((x.dot(w)) ** 2) / m

def df_debug(x,w,epsilon=1e-4):
    res = np.empty(len(w))
    for i in range(len(w)):
        w1 = w.copy()
        w1[i] += epsilon
        w2 = w.copy()
        w2[i] -= epsilon
        res[i] = (f(x,w1) - f(x,w2)) / 2 * epsilon
    return res

def df_math(x,w):
    return x.T.dot(x.dot(w)) * 2 / m

def direction(w):
    return w / np.linalg.norm(w)

def demean(x):
    return x - np.mean(x,axis=0)

def gradient_ascent(df,x,w,eta=0.001,n_iters=1e4,epsilon=1e-8):
    w = direction(w)
    cnt = 0
    while cnt < n_iters:
        cnt += 1
        last_w = w
        w += eta * df(x,w)
        w = direction(w)
        if np.abs(f(x,w)-f(x,last_w)) < epsilon:
            break
    return w

if __name__ == '__main__':
    w = np.random.random(size=n)
    res = gradient_ascent(df_debug,x,w) #res就是该样本的第一主成分
    print(res)
    x = demean(x)
    plt.scatter(x[:,0],x[:,1])
    plt.plot([0,res[0]*30],[0,res[1]*30],c='r')
    plt.show()
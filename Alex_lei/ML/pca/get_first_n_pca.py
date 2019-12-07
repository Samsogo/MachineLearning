'''

求一个样本的前n个主成分
第二主成分就是使得样本在第一主成分上没有分量，也就是用样本减去第一主成分

'''



import numpy as np
import matplotlib.pyplot as plt


m = 100
n = 2
x = np.empty([m,n])
x[:,0] = np.random.uniform(0,100,size=m)
x[:,1] = 0.8 * x[:,0] +3 + np.random.normal(0,10,size=m)

def f(x,w):
    return np.sum((x.dot(w)) ** 2) / m

def df(x,w):
    return x.T.dot(x.dot(w)) * 2 / m

def direction(w):
    return w / np.linalg.norm(w)

def demean(x):
    return x - np.mean(x,axis=0)

#求每个主成分
def first_compents(x, w, eta=0.001, n_iters=10000, epsilon=1e-8):
    w = direction(w)
    cnt = 0
    while True:
        cnt += 1
        last_w = w
        w = w+ eta * df(x, w)
        w = direction(w)
        if abs(f(x,last_w) - f(x,w)) < epsilon:
            break

    return w

#求前n个主成分
def first_n_compents(n, x):
    x_pca = x.copy()
    res = []
    for i in range(n):
        w = np.random.random(size=n)
        res_w = first_compents(x_pca,w)
        res.append(res_w)
        x_pca = x_pca - x_pca.dot(res_w).reshape(-1,1) * res_w

    return res

if __name__ == '__main__':
    #由于我们的样本只有两个样本，所以最多两个主成分，而且是相互垂直的
    x = demean(x)
    res = first_n_compents(2,x)
    print(res)
    print(res[0].dot(res[1]))
    plt.scatter(x[:,0],x[:,1])
    plt.plot([0,res[0][0]*30],[0,res[0][1]*30],c='r')
    plt.plot([0,res[1][0]*30],[0,res[1][1]*30],c='g')
    plt.show()

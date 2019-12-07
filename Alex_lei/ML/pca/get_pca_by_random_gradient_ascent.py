import  numpy as np
import matplotlib.pyplot as plt

m = 1000
n = 2
x = np.empty([m,n])
x[:,0] = np.random.uniform(0,100,size=m)
x[:,1] = 0.75 * x[:,0] + 3 + np.random.normal(size=m)

def f(x,w):
    return np.sum((x.dot(w)) ** 2) / m

def df_math(x,w):
    return x.T.dot(x.dot(w)) * 2 / len(x)

def df_dubug(x,w,epsilon):
    res = np.empty([1,n])
    for i in range(len(res)):
        w1 = w.copy()
        w1[i] += epsilon
        w2 = w.copy()
        w2[i] -= epsilon
        res[i] = (f(x,w1) - f(x,w2)) / (2*epsilon)
    return res

def direction(w):
    return w / np.linalg.norm(w)

def demean(x):
    return x - np.mean(x,axis=0)

def random_gradient_ascent(x,w,n_iters,t0,t1):
    w = direction(w)

    def learn_rate(cnt):
        return t0 / (cnt + t1)

    for i in range(n_iters):
        index = np.random.permutation(m)
        new_x = x[index]
        for j in range(len(new_x)):
            eta = learn_rate(i*n_iters + j)
            w = eta*df_math(new_x[i],w)
            w = direction(w)
    return w

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
    x = demean(x)
    res_w = random_gradient_ascent(x,w,3,5,50)
    print(res_w)
    res_w1 = gradient_ascent(df_math,x,w)
    print(res_w1)
    plt.scatter(x[:,0],x[:,1])
    plt.plot([0,res_w[0]*30],[0,res_w[0]*30],c='r') #随机梯度上升法的结果
    plt.plot([0,res_w1[0]*10],[0,res_w1[0]*10],c='g') #批量梯度上升法的结果
    plt.show()
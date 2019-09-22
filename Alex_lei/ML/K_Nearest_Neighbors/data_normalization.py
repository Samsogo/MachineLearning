'''
    数据归一化（标准化）处理是数据挖掘的一项基础工作。
    同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，
    为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。
    原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。
'''

'''
分为两类：
    最值归一化：适用于有边界的训练集
    均值方差归一化：有无边界都适用
'''

'''
    最值归一化(x-x(min))/(x(max)-x(min)),针对于每一列进行计算
    均值方差归一化，(x-x(mean))/x(std)，针对于每一列进行计算
'''


import numpy as np
import matplotlib.pyplot as plot

X = np.random.randint(0,100,(50,2))
arr = np.array(X,dtype=float)

#最值均一化
for i in range(0,arr.shape[1]):
    arr[:,i] = (arr[:,i] - np.min(arr[:,i])) / (np.max(arr[:,i]) - np.min(arr[:,i]))

print(arr)
plot.scatter(arr[:,0],arr[:,1],color='r')
plot.show()

#均值方差归一化

for i in range(0,arr.shape[1]):
    arr[:,i] = (arr[:,i] - np.mean(arr[:,i]))/np.std(arr[:,i])

print(arr)
plot.scatter(arr[:,0],arr[:,1],color='b')
plot.show()




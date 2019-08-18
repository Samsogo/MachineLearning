import numpy as np

x = np.random.normal(0,1,size=1000000)
print(x)

print(np.min(x)) # 返回最小值
index_min = np.argmin(x) #返回最小值的下标
print(x[index_min])

print(np.max(x))
index_max = np.argmax(x)
print(x[index_max])


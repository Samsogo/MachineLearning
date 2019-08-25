import numpy as np

x = np.arange(16)
np.random.shuffle(x)
print(x)

print(x[2]) #获取某个元素的值
print(x[1:4]) #切片
print(x[2:6:2]) #指定间距的切片

index = [2,4,7,9]  #索引数组
print(x[index]) #获取索引数组中的值

arr = np.array([[0,2],[1,4]])  #索引二维数组
print(x[arr]) #获取索引二维数组中元素的值

X = x.reshape(4,-1)
print(X)
print(X[1:3,:-1])

ind1 = np.array([1,3]) #行的索引
ind2 = np.array([2,0]) #列的索引
print(X[ind1,ind2])

print(X[:-2,ind2])

bool_index = [True,False,True,False]
print(X[:-1,bool_index])
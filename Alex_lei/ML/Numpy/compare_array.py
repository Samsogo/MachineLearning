import numpy as np

x = np.arange(16)
print(x)

print(x<3) #返回的是bool数组

print(x==3)

print(x!=3)

print(x*4 == 24-4*x)

print(x+1) #对x的每一个元素加1

print(x*2)

print(x/4)

print(np.sum(x<3)) #返回小于3的个数

print(np.any(x==0)) #只要向量中有等于0的就返回true

print(np.all(x==0)) #只有向量中全部等于0才返回true

print(x[x<5]) #x<5是bool数组，我们取的是true的元素的值

#-----------二维的同样支持

X = x.reshape(4,-1)
print(X)

print(X<3)

print(X==3)

print(np.sum(X<4))

print(np.count_nonzero(X<5))

print(np.any(X==0))

print(np.all(X==0))

print(np.sum(X<4,axis=1))

print(np.sum((X>3)&(X<5)))

print(np.sum(~(X==0)))

print(X[X[:,3]%3==0,:])

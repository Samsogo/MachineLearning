import numpy as np

arr1 = 5 - np.arange(1,13).reshape(4,3)
print(arr1)

#初始化一个4*3的矩阵，元素是随机的
arr2 = np.random.randint(1,10,size=12).reshape(4,3)
print(arr2)

print(arr2**2) #打印每个元素的平方

print(np.sqrt(arr2)) # 打印arr2的每个元素的平方根

print(np.exp(arr1)) #打印每个元素的指数值

print(np.log(arr2)) #打印每个元素的自然对数值

print(np.abs(arr1)) #打印每个元素的绝对值

print(arr1 + arr2) #相同形状的矩阵元素相加

print(arr1 - arr2) #相同形状的矩阵元素相减

print(arr2 * arr1) #矩阵相乘每一个元素对应相乘

print(arr1 / arr2) #矩阵相除

print(arr1 // arr2) #整除

print(arr1 % arr2) #取余


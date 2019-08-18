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


#------------统计运算函数----------

print(np.sum(arr1))  #将元素全部相加
print(np.sum(arr1,axis=0)) #将每一列求和
print(np.sum(arr1,axis=1)) #将每一行求和
print(np.max(arr1)) #求一个矩阵的最大值
print(np.max(arr1,axis=0)) #求每一列的最大值
print(np.max(arr1,axis=1)) #求每一行的最大值
print(np.min(arr1)) #求矩阵的最小值
print(np.min(arr1,axis=0)) #求矩阵的每一列的最小值
print(np.min(arr1,axis=1)) #求矩阵的每一行的最小值
print(np.cumsum(arr1)) #按从左往右，从上到下的顺序，对每个元素累积求和
print(np.cumsum(arr1,axis=0)) #计算每一列的累积和，并返回二维数组
print(np.cumsum(arr1,axis=1)) #计算每一行的累积和，并返回二维数组
print(np.mean(arr1)) #求矩阵的均值
print(np.mean(arr1,axis=0)) #求每一列的均值
print(np.mean(arr1,axis=1)) #求每一行的均值
print(np.median(arr1)) #求矩阵的中位数
print(np.median(arr1,axis=0)) #求每一列的中位数
print(np.median(arr1,axis=1)) #求每一行的中位数
print(np.var(arr1)) #求矩阵的方差
print(np.var(arr1,axis=0)) #求每一列的方差
print(np.var(arr1,axis=1)) #求每一行的方差
print(np.std(arr1)) #求矩阵的标准差
print(np.std(arr1,axis=0)) #求每一列的标准差
print(np.std(arr1,axis=1)) #求每一行的标准差


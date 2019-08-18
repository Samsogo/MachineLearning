import numpy as np

arr = np.array([[1,2,3],[6,8,9]])

print(arr.shape) #获取数组的行数和列数，以tuple的形式，打印结果为(2,3)
print(arr.dtype) #获取数组的元素类型，打印结果为int64

print(arr.ravel()) #将多维数组变为一维数组，打印结果为[ 1  2  3  6 8 9]
print(arr.flatten()) #同上
# 二者区别是ravels是生成的视图，他的变化会影响到原数组的变化，而flatten不会

print(arr.ndim) #输出数组的维数，打印结果为2
print(arr.size) #输出数组的大小，打印结果为6
print(arr.T) #将数组转置

#若元素是复数，可以打印其属性imag和real
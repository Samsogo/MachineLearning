import numpy as np

arr1 = np.array([1,2,3,4])
arr2 = np.array([2,3,4,5])

arr3 = np.hstack((arr1,arr2)) #将两个矩阵横向连接,必须满足行数相同
print(arr3)

arr4 = np.vstack((arr1,arr2)) #纵向连接两个矩阵,必须满足列数相同
print(arr4)

#重新分配矩阵的维数
l1 = np.array(np.arange(24))
print(l1.reshape(4,6))
print(l1)

#重新分配矩阵的维数，并直接改变原数组的形状
l1.resize(4,6)
print(l1)
print(l1.tolist()) # 将数组转化为列表，但不会改变原矩阵

#改变矩阵的类型
l2 = l1.astype(float)
print(l2)


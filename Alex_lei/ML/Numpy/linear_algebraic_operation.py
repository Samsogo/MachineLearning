import numpy as np

arr = np.array([[1,2,3,5],[2,4,1,6],[1,1,4,3],[2,5,4,1]])
print(arr)

print(np.linalg.det(arr)) #矩阵的行列式
print(np.linalg.inv(arr)) #矩阵的逆矩阵
print(np.trace(arr)) #矩阵的迹，也就是对角线元素的和
print(np.linalg.eig(arr)) #返回由特征根和特征向量组成的元组
print(np.linalg.qr(arr)) #返回矩阵的QR分解
print(np.linalg.svd(arr)) #返回矩阵的奇异值分解
print(np.dot(arr,arr)) #方阵的真正乘积运算
print(np.linalg.solve(arr,arr)) #求解线性方程组AX=B
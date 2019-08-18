import numpy as np

'''
1）求和sum

（2）最大值max

（3）最小值min

（4）平均值average

（5）中位数median

（6）prod，对所有的元素进行乘积

（7）percentile：百分比

（8）var：方差

（9）std：标准差
'''
arr = np.random.random(100000)
print(arr)

print(np.sum(arr))

print(np.max(arr))

print(np.min(arr))

print(np.average(arr))

print(np.median(arr))

print(np.prod(arr))

print(np.percentile(arr))

print(np.var(arr))

print(np.std(arr))
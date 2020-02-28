import numpy as np
from math import fabs
from sklearn import datasets
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库

radius=0.8#寻找邻居的半径
order_para=0#描述聚类情况，接近1时可结束

#加载数据集，是一个字典类似Java中的map
lris_df = datasets.load_iris()

#步骤一获取数据
dataset = np.array(lris_df.data[:-1])
N = len(dataset)

#步骤二、三
while fabs(order_para - 1) > 1e-2:
    #euclidean代表欧式距离
    distA=pdist(dataset,metric='euclidean')
    # 将distA数组变成一个矩阵,得到各点间的距离方阵
    distB = squareform(distA)
    
    #距离大于半径的全部置0
    np.putmask(distB, distB > radius, 0)
    np.putmask(distB, distB > 0, 1)
    
    #邻居个数
    nums = np.sum(distB,axis=1)
    
    #获取邻居坐标
    index = np.argwhere(distB > 0)
    
    #获取邻居矩阵,非邻居坐标用原坐标表示，方便计算
    neighbors = dataset.repeat(N,axis=0).reshape((N,N,4))
    for i in index:
        for j in range(4):
            neighbors[i[0],i[1],j] = dataset[i[1],j]
    
    #计算order_para
    res = 0
    for i in range(N):
        res += np.sum(np.exp(np.linalg.norm(np.subtract(neighbors[i],dataset[i]))*(-1)))
    order_para = res / N
    
    #计算sin
    for i in range(N):
        if nums[i]!=0:
            dataset[i] = dataset[i] + np.sum(np.sin(np.subtract(neighbors[i],dataset[i])),axis=0, keepdims=True) / nums[i]
print(dataset)
plt.scatter(dataset[:,0],dataset[:,1],alpha=0.6)
plt.show()
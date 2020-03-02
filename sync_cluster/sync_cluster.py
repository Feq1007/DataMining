import numpy as np
from math import fabs
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

import numpy as np  # 数组相关的库
import matplotlib.pyplot as plt  # 绘图库

radius=0.05#寻找邻居的半径
order_para=0#描述聚类情况，接近1时可结束

#加载数据集，是一个字典类似Java中的map
iris = datasets.load_iris()

#步骤一获取数据并归一化化处理
dataset = Normalizer().fit_transform(iris.data)
N = len(dataset)

#步骤二、三
while 1 - order_para > 1e-5:
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
    check = 0
    for i in range(N):
        check += np.sum(np.exp(np.linalg.norm(np.subtract(neighbors[i],dataset[i]))*(-1)))
    print(check)
    order_para = check / N
    
    #计算sin
    res = np.zeros(shape=(N,4))
    for i in range(N):
        if nums[i]!=0:
            res[i] = dataset[i] + np.sum(np.sin(np.subtract(neighbors[i],dataset[i])),axis=0, keepdims=True) / nums[i]
    dataset = res
    
    plt.scatter((dataset[:,0]+dataset[:,1])/2,(dataset[:,2]+dataset[:,3])/2,alpha=0.1,marker='o',norm=0.91)
    plt.show()

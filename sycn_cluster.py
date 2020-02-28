import numpy as np
from sklearn import datasets
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#加载数据集，是一个字典类似Java中的map
lris_df = datasets.load_iris()

radius=1#寻找领居的半径
order_para=0#描述聚类情况，接近1时可结束

#步骤一获取数据
dataset = np.array(lris_df.data[:-1])
length = len(dataset)

#步骤二、三
#euclidean代表欧式距离
distA=pdist(dataset,metric='euclidean')
# 将distA数组变成一个矩阵,得到各点间的距离
distB = squareform(distA)

#距离大于半径的全部置0
np.putmask(distB, distB > radius, 0)
np.putmask(distB, distB > 0, 1)
#邻居个数
nums = np.sum(distB,axis=1)

#获取邻居坐标
index = np.argwhere(distB > 0)

#获取邻居矩阵
neighbors = np.zeros(shape=(len(dataset),len(dataset),4))
for i in range(len(dataset)):
    for j in range(len(dataset)):
        for k in range(4):
            neighbors[i,j,k]=dataset[i,k]

for i in index:
    for j in range(4):
        neighbors[i[0],i[1],j] = dataset[i[1],j]

#计算
direc = [[]]
for i in range(len(dataset)):
    dataset[i] = dataset[i] + np.sum(np.sin(np.subtract(neighbors[i],dataset[i])),axis=0, keepdims=True)
print(dataset)

for i in range(N):
    print()
#neighbors = np.sum(np.sin(neighbors-))



    #计算邻居模长
    
    #更新dataset
    #dataset = dataset + 1/模长*np.sum()
    
    #计算order_para
    
    
#计算
#length = np.linalg.norm(distB,ord=2,axis=1,keepdims=True)

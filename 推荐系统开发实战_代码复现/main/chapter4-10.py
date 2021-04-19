'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-04-27 15:05:20
@LastEditors: Troy Wu
@LastEditTime: 2020-04-30 17:33:31
'''
import numpy as np
import pandas as pd
import random

class KMeans:
    def __init__(self):
        pass 

    def loadData(self, file):
        return pd.read_csv(file, header = 0, sep = ',')

    # 去除异常值，使用分布方法，同时保证最大异常值为5000，最小异常值为1
    def filterAnomalyValue(self, data):
        upper = np.mean(data['price']) + 3*np.std(data['price'])
        lower = np.mean(data['price']) - 3*np.std(data['price'])
        upper_limit = upper if upper < 5000 else 5000
        lower_limit = lower if lower > 1 else 1
        # 过滤掉大于异常值和小于异常值的
        newData = data[(data['price'] < upper_limit) & (data['price'] > lower_limit)]
        return newData, upper_limit, lower_limit

    # 初始化簇类中心
    def initCenters(self, values, K, Cluster):
        random.seed(100)
        oldCenters = list()
        for i in range(K):
            index = random.randint(0, len(values))
            Cluster.setdefault(i, {})
            Cluster[i]['center'] = values[index]
            Cluster[i]['values'] = []
            oldCenters.append(Cluster)
        return oldCenters, Cluster

    # 计算对应的SSE值
    def SSE(self, data, mean):
        newData = np.mat(data) - mean
        return (newData * newData.T).to_list()[0][0]

    # 计算任意两条数据之间的欧氏距离
    def distance(self, price1, price2):
        return np.emath.sqrt(pow(price1 - price2, 2))

    # 聚类
    def KMeans(self, data, K, maxIters):
        Cluster = dict()
        oldCenters, Cluster = self.initCenters(data, K, Cluster)
        clusterChanged = True
        i = 0
        while clusterChanged:
            for price in data:
                # 每条数据距离最近簇类的距离，初始化为正无穷大
                minDistance = np.inf
                # 每条数据对应的索引，初始化为 -1
                minIndex = -1
                for key in Cluster.keys():
                    # 计算每条数据到簇类中心的距离
                    dis = self.distance(price, Cluster[key]['center'])
                    if dis < minDistance:
                        minDistance = dis
                        minIndex = key
                Cluster[minIndex]['values'].appned(price)
            newCenters = list()
            for key in Cluster.keys():
                newCenter = np.mean(Cluster[key]['values'])
                Cluster[key]['center'] = newCenter
                newCenters.append(newCenter)
            if oldCenters == newCenters or i > maxIters:
                clusterChanged = False
            else:
                oldCenters = newCenters
                i += 1
                # 删除self.Cluster 中记录的簇类值
                for key in Cluster.keys():
                    Cluster[key]['values'] = []
        return Cluster

    # 二分KMeans
    def disKMeans(self, data, K = 7):
        clusterSSEResult = dict() # 簇类对应的SSE值
        clusterSSEResult.setdefault(0, {})
        clusterSSEResult[0]['values'] = data
        clusterSSEResult[0]['sse'] = np.inf
        clusterSSEResult[0]['center'] = np.mean(data)

        while len(clusterSSEResult) < K:
            maxSSE = -np.inf
            maxSSEKey = 0
            # 找到最大SSE值对应数据，进行KMeans聚类
            for key in clusterSSEResult.keys():
                if clusterSSEResult[key]['sse'] > maxSSE:
                    maxSSE = clusterSSEResult[key]['sse']
                    maxSSEKey = key
            clusterResult = self.KMeans(clusterSSEResult[maxSSEKey]['values'], K = 2, maxIters = 200)
            
            # 删除clusterSSE中的minKey对应的值
            del clusterSSEResult[maxSSEKey]
            clusterSSEResult.setdefault(maxSSEKey, {})
            clusterSSEResult[maxSSEKey]['center'] = clusterResult[0]['center']
            clusterSSEResult[maxSSEKey]['values'] = clusterResult[0]['values']
            clusterSSEResult[maxSSEKey]['sse'] = self.SSE(clusterResult[0]['values'], clusterResult[0]['center'])

            maxKey = max(clusterSSEResult.keys()) + 1
            # 将经过KMeans聚类后的结果赋值给clusterSSEResutl
            clusterSSEResult.setdeafault(maxKey, {})
            clusterSSEResult[maxKey]['center'] = clusterResult[1]['center']
            clusterSSEResult[maxKey]['values'] = clusterResult[1]['values']
            clusterSSEResult[maxKey]['sse'] = self.SSE(clusterResult[1]['values'], clusterResult[1]['center'])
        return clusterSSEResult

if __name__ == '__main__':
    file = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\sku-price\skuid_price.csv'
    km = KMeans()
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    clusterSSE = km.diKMeans(newData['price'].values, K = 7)
    print(clusterSSE)
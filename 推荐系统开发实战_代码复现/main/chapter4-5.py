'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-04-20 22:23:48
@LastEditors: Troy Wu
@LastEditTime: 2020-04-22 11:26:40
'''
import numpy as np

class KNN:
    def __init__(self, k):
        # k 为最近邻个数
        self.k = k

    # 将数据进行 min_max 标准化
    def Normalization(self, data):
        maxs = np.max(data, axis = 0)
        mins = np.min(data, axis = 0)
        new_data = (data - mins) / (maxs - mins)
        return new_data, maxs, mins

    # 计算k近邻
    def classify(self, one, data, labels):
        # 计算新样本与数据集中每个样本之间的距离，这里距离采用的是欧式距离计算方法
        differenceData = data - one
        squareData = (differenceData ** 2).sum(axis = 1)
        distance = squareData ** 0.5
        sortDistanceIndex = distance.argsort()
        # 统计k近邻的label
        labelCount = dict()
        for i in range(self.k):
            label = labels[sortDistanceIndex[i]]
            labelCount.setdefault(label, 0)
            labelCount[label] += 1
        # 计算结果
        sortLabelCount = sorted(labelCount.items(), key = lambda x: x[1], reverse = True)
        print(sortLabelCount)
        return sortLabelCount[0][0]

if __name__ == '__main__':
    #初始化类对象
    knn = KNN(3)
    features = np.array([[180, 76], [158, 43], [176, 78], [161, 49]])
    labels = ["男", "女", "男", "女"]
    newData, maxs, mins = knn.Normalization(features)
    one = np.array([176, 76])
    new_one = (one - mins) / (maxs - mins)
    result = knn.classify(new_one, newData, labels)
    print('数据{}的预测性别为{}'.format(one, result))
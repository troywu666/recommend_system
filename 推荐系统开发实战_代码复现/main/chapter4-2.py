'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-04-20 14:15:08
@LastEditors: Troy Wu
@LastEditTime: 2020-04-20 17:17:41
'''
import numpy as np
import math

class DiscreteByEntropy:
    '''
        group：最大分组数
        threshold：停止划分的最小熵
        result：保留最终划分结果
    '''
    def __init__(self, group, threshold):
        self.maxGroup = group
        self.minInfoThreshold = threshold
        self.result = dict()

    # 计算数据指定数据分组后的香农熵
    def _calEntropy(self, data):
        numData = len(data)
        labelCounts = {}
        for feature in data:
            # 获得标签
            oneLabel = feature[-1]
            # 如果标签不在新定义的字典里则创建该标签值
            labelCounts.setdefault(oneLabel, 0)
            # 该标签下含有数据的个数
            labelCounts[oneLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            # 同类标签出现的概率
            prob = float(labelCounts[key]) / numData
            # 以2为底求对数
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    # 按照调和信息熵最小化原则分割数据集
    def _split(self, data):
        # inf为正无穷大
        minEntropy = np.inf
        # 记录最终分割索引
        index = -1
        # 按照第一列对数据进行升序排序
        sortData = data[np.argsort(data[: , 0])]
        # 初始化最终分割数据后的熵
        lastE1, lastE2 = -1, -1
        # 返回的数据结构，包含数据和对应的熵
        S1 = dict()
        S2 = dict()
        for i in range(len(sortData)):
            # 分割数据集
            splitData1, splitData2 = sortData[: i+1], sortData[i+1 :]
            entropy1, entropy2 = self._calEntropy(splitData1), self._calEntropy(splitData2)
            entropy = entropy1*len(splitData1)/len(sortData) + entropy2*len(splitData2)/len(sortData)
            # 如果调和平均熵小于最小值
            if entropy < minEntropy:
                minEntropy = entropy
                index = i
                lastE1, lastE2 = entropy1, entropy2
        S1['entropy'] = lastE1
        S1['data'] = sortData[: index+1]
        S2['entropy'] = lastE2
        S2['data'] = sortData[index+1 :]
        return S1, S2, minEntropy

    # 对数据进行分组
    def train(self, data):
        # 需要遍历的key
        needSplitKey = [0]
        # 将整个数据作为一组
        self.result.setdefault(0, {})
        self.result[0]['entropy'] = np.inf
        self.result[0]['data'] = data
        group = 1
        for key in needSplitKey:
            S1, S2, entropy = self._split(self.result[key]['data'])
            if entropy > self.minInfoThreshold and group < self.maxGroup:
                self.result[key] = S1
                newKey = max(self.result.keys()) + 1
                self.result[newKey] = S2
                needSplitKey.extend([key])
                needSplitKey.extend([newKey])
                group += 1
            else:
                break

if __name__ =='__main__':
    dbe = DiscreteByEntropy(group = 6, threshold = 0.5)
    data = np.array(
            [
                [56, 1],   [87, 1],  [129, 0],  [23, 0],   [342, 1],
                [641, 1],  [63, 0],  [2764, 1], [2323, 0], [453, 1],
                [10, 1],   [9, 0],   [88, 1],   [222, 0],  [97, 0],
                [2398, 1], [592, 1], [561, 1],  [764, 0],  [121, 1],
            ]
        )
    dbe.train(data)
    print('result is {}'.format(dbe.result))
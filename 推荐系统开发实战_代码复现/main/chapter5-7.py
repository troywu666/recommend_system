'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-07 08:57:25
LastEditors: Troy Wu
LastEditTime: 2020-09-28 10:37:56
'''
import random
import math
import os
import json

class ItemCFRec:
    def __init__(self, datafile, ratio):
        self.datafile = datafile
        # 测试集和训练集的比例
        self.ratio = ratio
        self.data = self.loadData()
        self.trainData, self.testData = self.splitData(3, 47)
        self.items_sim = self.ItemSimilarityBest()

    def loadData(self):
        data = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split('::')
            data.append((userid, itemid, int(record)))
        return data
    
    def splitData(self, k, seed, M = 9):
        train, test = {}, {}
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record
        return train, test

    # 计算物品之间的相似度
    def ItemSimilarityBest(self):
        if os.path.exists('data/item_sim.json'):
            itemSim = json.load(open('data/item_sim.json', 'r'))
        else:
            itemSim = dict()
            item_user_count = dict()
            count = dict()
            for user, item in self.trainData.items():
                for i in item.keys():
                    item_user_count.setdefault(i, 0)
                    if self.trainData[str(user)][i] > 0:
                        item_user_count[i] += 1
                    for j in item.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if i != j and self.trainData[str(user)][i] > 0 and self.trainData[str(user)][j] > 0:
                            count[i][j] += 1
            # 共现矩阵 -> 相似矩阵
            for i, related_items in count.items():
                itemSim.setdefault(i, dict())
                for j, cuv in related_items.items():
                    itemSim[i].setdefault(j, 0)
                    itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])
        json.dump(itemSim, open('data/item_sim.json', 'w'))
        return itemSim

    def recommend(self, userA, k = 8, nitems = 40):
        '''
            为用户进行推荐
                user：用户
                k：k个临近产品
                nitems：共返回n个产品
        '''
        result = dict()
        u_items = self.trainData.get(userA, {})
        for i, pi in u_items.items():
            for j, wj in sorted(self.items_sim[i].items(), key = lambda x: x[1], reverse = True)[: k]:
                if i == j:
                    continue
                result.setdefault(j, 0)
                result[j] += wj * pi
        return dict(sorted(result.items(), key = lambda x: x[1], reverse = True)[: nitems])
    
    def precision(self, k = 8, nitems = 40):
        hit = 0
        precision = 0
        for userA in self.testData.keys():
            u_items = self.testData.get(userA, {})
            result = self.recommend(userA, k = k, nitems = nitems)
            for item, rate in result.items():
                if item in u_items:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)

if __name__ == '__main__':
    ib = ItemCFRec(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\ml-1m\ratings.dat', [1, 9])
    print(ib.precision())
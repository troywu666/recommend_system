'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-04 16:56:23
@LastEditors: Troy Wu
@LastEditTime: 2020-05-07 00:13:39
'''
import math

class ItemCF:
    def __init__(self):
        self.user_score_dict = self.initUserScore()
        self.items_sim = self.ItemSimilarityBest()

    # 初始化用户评分数据
    def initUserScore(self):
        user_score_dict = {
            "A": {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0},
            "B": {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5},
            "C": {"a": 0.0, "b": 3.5, "c": 0.0, "d": 0.0, "e": 3.0},
            "D": {"a": 0.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 3.0},
        }
        return user_score_dict

    # 计算item之间的相似度
    def ItemSimilarity(self):
        itemSim = dict()
        # 得到每个物品有多少用户产生过行为
        item_user_count = dict()
        # 共现矩阵
        count = dict()
        for user, item in self.user_score_dcit.items():
            for i in item.keys():
                item_user_count.setdefault(i, 0)
                if self.user_score_dict[user][i] > 0.0:
                    item_user_count[i] += 1
                for j in items.keys():
                    count.setdefault(i, {}).setdefault(j, 0)
                    if i != j and self.user_score_dict[user][i] > 0.0 and self.user_score_dict[user][j] > 0.0:
                        count[i][j] += 1
        # 共现矩阵 -> 相似度矩阵
        for i, related_items in count.items():
            itemSim.setdefault(i, dict())
            for j, cuv in related_items.items():
                itemSim[i].setdefault(j, 0)
                itemSim[i][j] = cuv / item_user_count[i]
        return itemSim

    # 计算item之间的相似度 优化后
    def ItemSimilarityBest(self):
        itemSim = dict()
        item_user_count = dict()
        count = dict()
        for user, item in self.user_score_dict.items():
            for i in item.keys():
                item_user_count.setdefault(i, 0)
                if self.user_score_dict[user][i] > 0.0:
                    item_user_count[i] += 1
                for j in item.keys():
                    count.setdefault(i, {}).setdefault(j, 0)
                    if i != j and self.user_score_dict[user][i] > 0.0 and self.user_score_dict[j] > 0.0:
                        count[i][j] += 1
        for i, related_items in count.items():
            itemSim.setdefault(i, {})
            for j, cuv in related_items.items():
                itemSim[i].setdefault(j, 0)
                itemSim[i][j] += cuv / math.sqrt(item_user_count[i] * item_user_count[j])
        return itemSim

    def preUserItemScore(self, userA, item):
        score = 0.0
        for item1 in self.items_sim[item].keys():
            if item1 != item:
                score += self.items_sim[item][item1] * self.user_score_dict[userA][item1]
        return score

    def recommend(self, userA):
        user_item_score_dict = dict()
        for item in self.user_score_dict[userA].keys():
            user_item_score_dict[item] = self.preUserItemScore(userA, item)
        return user_item_score_dict

if __name__ == '__main__':
    ib = ItemCF()
    print(ib.recommend('C'))
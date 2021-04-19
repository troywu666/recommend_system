'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-01 11:45:26
@LastEditors: Troy Wu
@LastEditTime: 2020-05-02 10:53:07
'''
import json
import pandas as pd
import numpy as np
import math
import random

class CBRecommend:
    # 加载dataProcessing.py中预处理的数据
    def __init__(self, K):
        self.K = K
        self.item_profile = json.load(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\item_profile.json', 'r')
        self.user_profile = json.load(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\user_profile.json', 'r')

    # 获取用户未进行评分的item列表
    def get_none_score_item(self, user):
        items = pd.read_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\movies.csv')
        data = pd.read_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\ratings.csv')
        have_score_items = data[data['UserID'] == user]['MovieID'].values
        none_score_items = set(items) - set(have_score_items)
        return none_score_items
    
    # 获取用户对item的喜好程度
    def cosUI(self, user, item):
        Uia = sum(np.array(self.user_profile[str(user)]) * np.array(self.item_profile[str(item)]))
        Ua = math.sqrt(sum([math.pow(one, 2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one, 2) for one in self.item_profile[str(item)]]))
        return Uia / (Ua * Ia)

    # 为用户进行电影推荐
    def recommend(self, user):
        user_result = {}
        item_list = self.get_none_score_item(user)
        for item in item_list:
            user_result[item] = self.cosUI(user, item)
        if self.K is None:
            result = sorted(user_result.items(), key = lambda k: k[1], reversed = True)
        else:
            result = sorted(user_result.items(), key = lambda k: k[1], reversed = True)[: self.K]
        print(result)
            
    # 推荐系统效果评估
    def evaluate(self):
        evas = []
        data = pd.read_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\5-chapter\data\ratings.csv')
        for user in random.sample([one for one in range(1, 6040)], 20):
            have_score_items = data[data['UserID'] == user]['MovieID'].values
            items = pd.read_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\movies.csv')
            user_result = {}
            for item in items:
                user_result[item] = self.cosUI(user, item)
            results = sorted(user_result, key = lambda x: x[1], reverse = True)[ : len(have_score_items)]
            rec_items = []
            for one in results:
                rec_items.append(one[0])
            eva = len(set(rec_items) & set(have_score_items)) / len(have_score_items)
            evas.append(eva)
        return sum(evas) / len(evas)

if __name__ == '__main__':
     cb = CBRecommend(K = 10)
     cb.recommend(1)
     print(cb.evaluate())

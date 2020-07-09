'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-04-30 10:17:54
@LastEditors: Troy Wu
@LastEditTime: 2020-05-01 14:12:05
'''
import pandas as pd
import json
import os

class DataProcessing:
    def __init__(self):
        pass

    def process(self):
        self.process_user_data()
        self.process_movies_data()
        self.process_rating_data()
    
    def process_user_data(self, file = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\ml-1m\users.dat'):
        if os.path.exists(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\5-chapter\data\users.csv'):
            print('user.csv已经存在')    
        fp = pd.read_table(file, sep = '::', engine = 'python', names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        fp.to_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\users.csv', index = False)

    def process_rating_data(self, file = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\ml-1m\ratings.dat'):
        if os.path.exists(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\5-chapter\data\ratings.csv'):
            print('ratings.csv已经存在')
        fp = pd.read_table(file, sep='::', engine='python',names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        fp.to_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\ratings.csv', index=False)

    def process_movies_date(self, file = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\ml-1m\movies.dat'):
        if os.path.exists(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\movies.csv'):
            print("movies.csv已经存在")
        fp = pd.read_table(file, sep='::', engine='python',names=['MovieID', 'Title', 'Genres'])
        fp.to_csv(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\movies.csv', index=False)

    # 获取item的特征信息矩阵
    def prepare_item_profile(self, file = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\5-chapter\data\movies.csv'):
        items = pd.read_csv(file)
        item_ids = set(items['MovieID'].values) 
        self.item_dict = {}
        genres_all = list()
        # 将每个电影的类型放在item_dict中
        for item in item_ids:
            genres = items[items['MovieID'] == item]['Genres'].values[0].split('|')
            self.item_dict.setdefault(item, []).extend(genres)
            genres_all.extend(genres)
        self.genres_all = set(genres_all)
        # 将每个电影的特征信息矩阵放在 self.item_matrix中
        # 保存dict时， key只能为str，所以这里对item_id 做str()转换
        self.item_matrix = {}
        for item in self.item_dict.keys():
            self.item_matrix[str(item)] = [0] * len(set(self.genres_all))
            for genre in self.item_dict[item]:
                index = list(set(genres_all)).index(genre)
                self.item_matrix[str(item)][index] = 1
        json.dump(self.item_matrix, open(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\item_profile.json'))
    
    # 计算用户偏好矩阵
    def prepare_user_profile(self, file = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\ratings.csv'):
        users = pd.read_csv(file)
        user_ids = set(users['UserID'].values)
        # 将users信息转化成dict
        users_rating_dict = {}
        for user in user_ids:
            users_ratings_dict.setdefault(user, {})
        with open(file, 'r') as fr:
            for line in fr.readlines():
                if not line.startswith('UserID'):
                    (user, item, rate) = line.split(',')[: 3]
                    users_rating_dict[user][item] = int(rate)

        # 获取用户对每个类型下都有哪些电影评了分
        self.user_matrix = {}
        # 遍历每个用户
        for user in users_rating_dict.keys():
            score_list = users_rating_dict[user].values()
            # 用户的平均打分
            avg = sum(score_list) / len(score_list)
            self.user_matrix[user] = []
            # 遍历每个类型（保证item_profile和user_profile信息矩阵中每列表示的类型一致）
            for genre in sefl.genres_all:
                score_all = 0.0
                score_len = 0
                # 遍历每个item
                for item in users_rating_dict[user].keys():
                    # 判断类型是否在用户评分过的电影里
                    if genre in self.item_dict[int(item)]:
                        score_all += (users_rating_dict[user][item] - avg)
                        score_len += 1
                if score_len == 0:
                    self.user_matrix[user].append(0.0)
                else:
                    self.user_matrix[user].append(score_all / score_len)
        json.dump(self.user_matrix, open(r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\main\user_profile.json', 'w'))

if __name__ == '__main__':
    dp = DataProcessing()
    dp.process()
    dp.prepare_item_profile()
    dp.prepare_user_profile()
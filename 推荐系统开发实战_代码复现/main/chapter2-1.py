'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-04-19 10:40:45
@LastEditors: Troy Wu
@LastEditTime: 2020-04-19 20:26:52
'''
import json
import random
import math
import os

class FirstRec:
    '''
        初始化函数
            file_path：原始文件路径
            seed：随机种子数
            k：选取的近邻客户个数
            n_items：为每个用户推荐的电影数
    '''
    def __init__(self, file_path, seed, k, n_items):
        self.file_path = file_path
        self.seed = seed
        self.users_1000 = self.__select_1000_users()
        self.k = k
        self.n_items = n_items
        self.train, self.test = self._load_and_split_data()
    
    # 获取所有用户并随机选取1000个
    def __select_1000_users(self):
        print('随机选取1000个用户！')
        if os.path.exists('data/train.json') and os.path.exists('data/test.json'):
            return list()
        else:
            users = set()
            # 获取所有用户
            for file in os.listdir(self.file_path):
                one_path = '{}/{}'.format(self.file_path, file)
                print('{}'.format(one_path))
                with open(one_path, 'r') as fp:
                    for line in fp.readlines():
                        if line.strip().endswith(':'):
                            continue
                        userID, _, _ = line.split(',')
                        users.add(userID)
            # 随机选取1000个
            users_1000 = random.sample(list(users), 1000)
            print(users_1000)
            return users_1000

    # 加载数据，并拆分成训练集和测试集
    def _load_and_split_data(self):
        train = dict()
        test = dict()
        if os.path.exists('data/train.json') and os.path.exists('data/test.json'):
            print('从文件中加载训练集和测试集')
            train = json.load(open('data/train.json'))
            test = json.load(open('data/test.json'))
            print('从文件中加载数据完成')
        else:
            # 设置产生随机数的种子，保证每次实验产生的随机结果都一致
            random.seed(self.seed)
            for file in os.listdir(self.file_path):
                one_path = '{}/{}'.format(self.file_path, file)
                print('{}'.format(one_path))
                with open(one_path, 'r') as fp:
                    movieID = fp.readline().split(':')[0]
                    for line in fp.readlines():
                        if line.endswith(':'):
                            continue
                        userID, rate, _ = line.split(',')
                        # 判断用户是否在所选择的1000个用户中
                        if userID in self.users_1000:
                            if random.randint(1, 50) == 1:
                                test.setdefault(userID, {})[movieID] = int(rate)
                            else:
                                train.setdefault(userID, {})[movieID] = int(rate)
            print('加载数据到 data/tarin.json 和 data/test.json')
            json.dump(train, open('data/train.json', 'w'))
            json.dump(test, open('data/test.json', 'w'))
            print('数据加载完成')
        return train, test   
    
    '''
        计算皮尔逊系数
            rating1：用户1的评分记录，形式如{"movieid1":rate1,"movieid2":rate2,...}
            rating2：用户2的评分记录，形式如{"movieid1":rate1,"movieid2":rate2,...}
    '''
    def pearson(self, rating1, rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        num = 0
        for key in rating1.keys():
            if key in rating2.keys():
                num += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += math.pow(x, 2)
                sum_y2 += math.pow(y, 2)
        if num == 0:
            return 0
        # 斯皮尔相关系数分母
        denominator = math.sqrt(sum_x2 - math.pow(sum_x, 2)/num) * math.sqrt(sum_y2 - math.pow(sum_7, 2)/num)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y)/num) / denominator

    '''
        用户userID进行电影推荐
            userID：用户ID
    '''
    def recommend(self, userID):
        neighborUser = dict()
        for user in self.train.keys():
            if userID != user:
                distance = self.pearson(self.train[userID], self.train[user])
                neighborUser[user] = distance
        # 字典排序
        newNU = sorted(neighborUser.items(), key = lambda x: x[1], reverse = True)
        movies = dict()
        for (sim_user, sim) in newNU[: self.k]:
            for movieID in self.train[sim_user].keys():
                movies.setdeafult(movieID, 0)
                movies[movieID] += sim * self.train[sim_user][movieID]
        newMovies = sorted(movies.items(), key = lambda x: x[1], reverse = True)
        return newMovies

    '''
        推荐系统效果评估函数
            num：随机抽取 num 个用户计算准确率
    '''
    def evaluate(self, num = 30):
        print('开始计算准确率')
        precisions = list()
        random.seed(10)
        for userID in random.sample(self.test.keys(), num):
            hit = 0
            result = self.recommend(userID)[n_items]
            for (item, rate) in result.items():
                if item in self.test[userID]:
                    hit += 1
            precisions.append(hit / self.n_items)
        return sum(precisions) / precisions.__len__()

# main函数，程序的入口
if __name__ == '__main__':
    file_path = r'D:\troywu666\personal_stuff\Practice\推荐系统\推荐系统开发实战_代码复现\代码\data\netflix\training_set'
    seed = 30
    k = 15
    n_items = 20
    f_rec = FirstRec(file_path, seed, k, n_items)
    # 计算用户 195100 和 1547579 的皮尔逊相关系数
    r = f_rec.pearson(f_rec.train['195100'], f_rec.train['1547579'])
    print("195100 和 1547579的皮尔逊相关系数为：{}".format(r))
    # 为用户 195100 进行电影推荐
    result = f_rec.recommend('195100')
    print("为用户ID为：195100的用户推荐的电影为：{}".format(result))
    print('算法的推荐准确率为{}'.format(f_rec.evaluate()))
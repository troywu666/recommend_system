'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-04-22 22:05:26
@LastEditors: Troy Wu
@LastEditTime: 2020-04-27 14:37:41
'''
import numpy as np

class NaiveBayesian:
    def __init__(self, alpha):
        self.classP = dict()
        self.classP_feature = dict()
        self.alpha = alpha # 平滑值

    # 计算高斯分布函数值
    def gaussian(self, mu, sigma, x):
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) *np.exp(-(x - mu)**2 / (2 * sigma**2)) 
    
    # 计算某个特征列对应的均值和标准差
    def _calMuAndSigma(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return (mu, sigma)

    def train(self, data, labels):
        numData = len(labels)
        numFeatures = len(data[0])
        # 是异常用户的概率
        self.classP[1] = ((sum(labels) + self.alpha)*1.0 / (numData + self.alpha*len(set(labels))))
        # 不是异常用户的概率
        self.classP[0] = 1 - self.classP[1]
        
        # 用来存放每个label下每个特征标签下对应的高斯分布中的均值和方差
        # {label1: {feature1: {mean: 0.2, var: 0.8}, feature2: {}}, label2: {}}
        self.classP_feature = dict()
        # 遍历每个标签
        for c in set(labels):
            self.classP_feture[c] = {}
            for i in range(numFeatures):
                feature = data[np.equal(labels, c)][ :, i]
                self.classP_feature[c][i] = self._calMuAndSigma(feature)
                
    # 预测新用户是否是异常用户
    def predict(self, x):
        label = -1 # 初始化类别
        maxP = 0

        # 遍历所有的label值
        for key in self.classP.keys():
            label_P = self.classP[key] 
            currentP = 1.0
            feature_P = self.classP_feature[key]
            j = 0
            for fp in feature_P.keys():
                currentP *= self.gaussian(feature_P[fp][0], feature_P[fp][1], x[j])
                j += 1
            # 如果计算出来的概率大于初始的最大概率，则进行最大概率赋值 和 对应的类别记录
            if currentP * label_P > maxP:
                maxP = currentP * label_P
                label = key
        return label                    

if __name__ == '__main__':
    nb = NaiveBayesian(1.0)
    data = np.array(
            [
                [320, 204, 198, 265],
                [253, 53, 15, 2243],
                [53, 32, 5, 325],
                [63, 50, 42, 98],
                [1302, 523, 202, 5430],
                [32, 22, 5, 143],
                [105, 85, 70, 322],
                [872, 730, 840, 2762],
                [16, 15, 13, 52],
                [92, 70, 21, 693],
            ]
        )
    labels = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    nb.train(data, labels)
    label = nb.predict(np.array([134, 84, 235, 349]))
    print('未知类型用户对应的行为数据为：[134,84,235,349]，该用户的可能类型为：{}'.format(label))
'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-05-11 17:59:57
@LastEditors: Troy Wu
@LastEditTime: 2020-05-12 10:01:11
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

class ChurnPredWithGBDTAndLR:
    def __init__(self):
        self.file = 'data/new_churn.csv'
        self.data = self.load_data()
        self.train, self.test = self.split()

    def load_data(self):
        return pd.read_csv(self.file)

    def split(self):
        train, test = train_test_split(self.data, test_size = 0.1, random_state = 40)
        return train, test

    def train_model(self):
        label = 'Churn'
        ID = 'customerID'
        x_columns = [x for x in self.train.columns if x not in [label, ID]]
        x_train = self.train[x_columns]
        y_train = self.train[label]

        gbdt = GradientBoostingClassifier()
        gbdt.fit(x_train, y_train)

        gbdt_lr = LogisticRegression()
        enc = OneHotEncoder()
        
        enc.fit(gbdt.apply(x_train).reshape(-1, 100))
        gbdt_lr.fit(enc.transform(gbdt.apply(x_train).reshape(-1, 100)), y_train)

        return enc, gbdt, gbdt_lr

    def evaluate(self, enc, gbdt, gbdt_lr):
        label = 'Churn'
        ID = 'customerID'
        x_columns = [x for x in self.test.columns if x not in [label, ID]]
        x_test = self.test[x_columns]
        y_test = self.test[label]

        gbdt_y_pred = gbdt.predict_proba(x_test)
        new_gbdt_y_pred = list()
        for y in gbdt_y_pred:
            new_gbdt_y_pred.append(1 if y[1] > 0.5 else 0)
        print(mean_squared_error(y_test, new_gbdt_y_pred))
        print(metrics.accuracy_score(y_test.values, new_gbdt_y_pred))
        print(metrics.roc_auc_score(y_test.values, new_gbdt_y_pred))
        
        gbdt_lr_y_pred = gbdt_lr.predict_proba(enc.transform(gbdt.apply(x_test).reshape(-1, 100)))
        new_gbdt_lr_y_pred = list()
        for y in gbdt_lr_y_pred:
            new_gbdt_lr_y_pred.append(1 if y[1] > 0.5 else 0)
        print(mean_squared_error(y_test, new_gbdt_lr_y_pred))
        print(metrics.accuracy_score(y_test.values, new_gbdt_lr_y_pred))
        print(metrics.roc_auc_score(y_test.values, new_gbdt_lr_y_pred))
        
if __name__ == '__main__':
    pred = ChurnPredWithGBDTAndLR()
    enc, gbdt, gbdt_lr = pred.train_model()
    pred.evaluate(enc, gbdt, gbdt_lr)
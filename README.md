__推荐系统由基础算法到深度学习的应用__  
参考
https://github.com/troywu666/recommend_system/blob/master/recommend_system.ipynb



# 电影推荐系统

标签：Tensorflow、矩阵分解、Surprise、PySpark

## 1、用Tensorflow实现矩阵分解

### 1.1、定义one_batch模块

```python
import numpy as np
import pandas as pd

def read_and_process(filename, sep = '::'):
    col_names = ['user', 'item', 'rate', 'timestamp']
    df = pd.read_csv(filename, sep = sep, header = None, names = col_names, 
                     engine = 'python')
    df['user'] -= 1
    df['item'] -= 1
    for col in ('user', 'item'):
        df[col] = df[col].astype(np.float32)
    df['rate'] = df['rate'].astype(np.float32)
    return df

def get_data():
    df = read_and_process("./movielens/ml-1m/ratings.dat", sep = '::')
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop = True)##打乱数据
    split_index = int(rows * 0.9)
    df_train = df[0: split_index]
    df_test = df[split_index:].reset_index(drop = True)
    print(df_train.shape, df_test.shape)
    return df_train, df_test
    
class ShuffleDataIterator(object):
    def __init__(self, inputs, batch_size = 10):
        ##注意这里的输入
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.len
    
    def __next__(self):
        return self.next()
    
    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]
       
class OneEpochDataIterator(ShuffleDataIterator):
    def __init__(self, inputs, batch_size=10):
        super(OneEpochDataIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), 
                                            np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0
    ##next函数不能写在__init__下面
    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]
```

### 1.2、构建优化部分

```python
def inference_svd(user_batch, item_batch, user_num, item_num, dim = 5, 
                  device = '/cpu:0'):
    with tf.device('/cpu:0'):
        global_bias = tf.get_variable('global_bias', shape = [])
        w_bias_user = tf.get_variable('embd_bias_user', shape = [user_num])
        w_bias_item = tf.get_variable('embd_bias_item', shape = [item_num])
        
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name = 'bias_user')
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name = 'bias_item')
        
        w_user = tf.get_variable('embd_user', shape = [user_num, dim], initializer = tf.truncated_normal_initializer(stddev = 0.02))
        w_item = tf.get_variable('embd_item', shape = [item_num, dim], initializer = tf.truncated_normal_initializer(stddev = 0.02))
        
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name = 'embedding_user')
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name = 'embedding_item')
    
    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)##tf.multiply是元素点乘
        infer = tf.add(infer, global_bias)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name = 'svd_inference')
        
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), 
                             name = 'svd_regularization')
    return infer, regularizer

def optimizer(infer, regularizer, rate_batch, learning_rate = 0.001, reg = 0.1, 
              device = '/cpu:0'):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype = tf.float32, shape = [], name = 'l2')
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(
            learning_rate).minimize(cost, global_step = global_step)
    return cost, train_op
```

### 1.3、定义训练函数

```python
import time 
from collections import deque
import numpy as np
import pandas as pd
from six import next
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

np.random.seed(12321)

batch_size = 2000
user_num = 6040
item_num = 3952
dim = 15
epoch_max = 200
device = '/cpu:0'

def make_scalar_summary(name, val):
    return summary_pb2.Summary(
        value = [summary_pb2.Summary.Value(tag = name, simple_value = val)])

def svd(train, test):
    samples_per_batch = len(train) // batch_size
    
    iter_train = ShuffleDataIterator(
        [train['user'], train['item'], train['rate']], 
        batch_size = batch_size)##注意iuputs
    iter_test = OneEpochDataIterator(
        [test['user'], test['item'], test['rate']], batch_size = -1)
    user_batch = tf.placeholder(tf.int32, shape = [None], name = 'id_user')
    item_batch = tf.placeholder(tf.int32, shape = [None], name = 'id_item')
    rate_batch = tf.placeholder(tf.float32, shape = [None])
    
    infer, regularizer = inference_svd(
        user_batch, item_batch, user_num = user_num, 
        item_num = item_num, dim = dim, device = device)
    global_step = tf.train.get_or_create_global_step()
    cost, train_op = optimizer(
        infer, regularizer, rate_batch, 
        learning_rate = 0.001, reg = 0.05, device = device)
    
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir = './data', graph = sess.graph)
        print('{} {} {} {}'.format('epoch', 'train_error', 'val_error', 'elapsed_time'))
        errors = deque(maxlen = samples_per_batch)
        start = time.time()
        for i in range(epoch_max * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run(
                [train_op, infer], feed_dict = {user_batch: users, 
                                                item_batch: items, rate_batch: rates})
            pred_batch = np.clip(pred_batch, 1.0, 5.0)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(
                        infer, feed_dict = {user_batch: users, item_batch: items})
                    pred_batch = np.clip(pred_batch, 1.0, 5.0)
                    test_err2 = np.append(test_err2, np.power((pred_batch - rates), 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print('{:3d} {:f} {:f} {:f}(s)'.format(
                    i // samples_per_batch, train_err, test_err, end - start))
                train_err_summary = make_scalar_summary('training_error', train_err)
                test_err_summary = make_scalar_summary('testing_error', test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end
```

## 2、使用surprise库实现电影推荐

```python
from surprise import KNNBaseline
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(line_format = 'user item rating timestamp', sep = '::')
data = Dataset.load_from_file('./movielens/ml-1m/ratings.dat', reader = reader)
algo = KNNBaseline()
perf = cross_validate(algo, data, measures = ['RMSE', 'MAE'], cv = 3, verbose = True)

with open('./movielens/ml-1m/movies.dat', 'r', encoding = 'ISO-8859-1') as f:
    movies_id_dic = {}
    id_movies_dic = {}
    for line in f.readlines():
        movies = line.strip().split('::')
        id_movies_dic[int(movies[0]) - 1] = movies[1]
        movies_id_dic[movies[1]] = int(movies[0]) - 1
     
toy_story_neighbors = algo.get_neighbors(movie_id, k = 5)
print('最接近《Toy Story (1995)》的5部电影是：')
for i in toy_story_neighbors:
    print(id_movies_dic[i])
```

## 3、用pyspark实现矩阵分解与预测

### 3.1、配置spark的运行环境

```python
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

conf = SparkConf().setMaster('local').setAppName('movielenALS').set('spark.excutor.memory', '2g')
sc = SparkContext.getOrCreate(conf)
```

### 3.2、将数据转换为RDD格式

```python
from pyspark.mllib.recommendation import Rating

ratings_data = sc.textFile('./movielens/ml-1m/ratings.dat')
ratings_int = ratings_data.map(lambda x: x.split('::')[0:3])
rates_data = ratings_int.map(lambda x: Rating(int(x[0]), int(x[1]), int(x[2])))
```

### 3.3、预测

```python
sc.setCheckpointDir('checkpoint/')
ALS.checkpointInterval = 2
model = ALS.train(ratings = rates_data, rank = 20, iterations = 5, lambda_ = 0.02)
```

预测user14对item25的评分

```python
print(model.predict(14, 25))
```

预测item25的最值得推荐的10个user

```python
print(model.recommendUsers(25, 10))
```

预测user14的最值得推荐的10个item

```python
print(model.recommendProducts(14, 10))
```

预测出每个user最值得被推荐的3个item

```python
print(model.recommendProductsForUsers(3).collect())
```
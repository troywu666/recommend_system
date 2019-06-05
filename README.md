
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#推荐系统" data-toc-modified-id="推荐系统-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>推荐系统</a></span><ul class="toc-item"><li><span><a href="#推荐系统的评价" data-toc-modified-id="推荐系统的评价-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>推荐系统的评价</a></span></li><li><span><a href="#基于内容的推荐" data-toc-modified-id="基于内容的推荐-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>基于内容的推荐</a></span><ul class="toc-item"><li><span><a href="#特征提取" data-toc-modified-id="特征提取-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>特征提取</a></span><ul class="toc-item"><li><span><a href="#结构化数据" data-toc-modified-id="结构化数据-1.2.1.1"><span class="toc-item-num">1.2.1.1&nbsp;&nbsp;</span>结构化数据</a></span></li><li><span><a href="#非结构化数据" data-toc-modified-id="非结构化数据-1.2.1.2"><span class="toc-item-num">1.2.1.2&nbsp;&nbsp;</span>非结构化数据</a></span></li></ul></li><li><span><a href="#用户偏好计算" data-toc-modified-id="用户偏好计算-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>用户偏好计算</a></span></li><li><span><a href="#内容召回" data-toc-modified-id="内容召回-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>内容召回</a></span></li><li><span><a href="#物品排序" data-toc-modified-id="物品排序-1.2.4"><span class="toc-item-num">1.2.4&nbsp;&nbsp;</span>物品排序</a></span></li></ul></li><li><span><a href="#基于领域的推荐（协同算法）" data-toc-modified-id="基于领域的推荐（协同算法）-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>基于领域的推荐（协同算法）</a></span><ul class="toc-item"><li><span><a href="#基于用户的协同过滤算法" data-toc-modified-id="基于用户的协同过滤算法-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>基于用户的协同过滤算法</a></span></li><li><span><a href="#基于物品的协同过滤算法" data-toc-modified-id="基于物品的协同过滤算法-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>基于物品的协同过滤算法</a></span><ul class="toc-item"><li><span><a href="#基于共同喜欢物品的用户列表计算（购买数）:" data-toc-modified-id="基于共同喜欢物品的用户列表计算（购买数）:-1.3.2.1"><span class="toc-item-num">1.3.2.1&nbsp;&nbsp;</span>基于共同喜欢物品的用户列表计算（购买数）:</a></span></li><li><span><a href="#基于余弦的相似度（评分数据）" data-toc-modified-id="基于余弦的相似度（评分数据）-1.3.2.2"><span class="toc-item-num">1.3.2.2&nbsp;&nbsp;</span>基于余弦的相似度（评分数据）</a></span></li><li><span><a href="#热门物品的惩罚" data-toc-modified-id="热门物品的惩罚-1.3.2.3"><span class="toc-item-num">1.3.2.3&nbsp;&nbsp;</span>热门物品的惩罚</a></span></li></ul></li></ul></li></ul></li></ul></div>

# 推荐系统

__推荐系统需同时具备：__
>速度快  
>准确度高

## 推荐系统的评价

好的推测不代表是好的推荐！如果本来用户就有了主观选择趋势，此时的预测实际上并不能提高潜在商品的被选择率。

- 评测方法：

>1、离线实验  
>缺点：较难得到离线的与商业指标相关的指标  

>2、用户调查  
>缺点：用户量大时，成本过高；用户量小时，得出的结论往往没有统计意义

>3、在线实验（较常使用A/B test）  
>缺点：周期较长，必须进行长期的实验才有可靠的结果

## 基于内容的推荐

__优点：__
>1、易于定位问题  
>2、能为具有特殊兴趣爱好的用户进行推荐  
>3、物品没有冷启动问题    

__缺点：__
>1、要求取出的特征必须具有良好的结构性  
>2、推荐精度低，相同内容特征的物品差异性不大

### 特征提取
#### 结构化数据
>可取用二进制的方法进行表示
#### 非结构化数据
>1、对于英文，可直接取词；对于中文，需先直接分词  
>2、统计方法分两种  
>>2.1、基础统计法：商品或词汇出现就取1，没出现就取0（当很多文章都包含某个词时，则这个词没有信息量）   
>><span style='color:red'>__2.2、词频统计法：__</span>  
    $$w_{k,j}=\frac{TF-IDF(t_{k},d_{j})}{\sqrt{\sum{TF-IDF(t_{k},d_{j})^{2}}}}$$  
$w_{k,j}$是 词k在文章j中的权重  
$$TF-IDF(t_{k},d_{j})=TF(t_{k},d_{j})\cdot\log\frac{N}{n_{k}}$$
$TF(t_{k},d_{j})$是词k在商品或文章j中出现的次数   
$n_{k}$是包含词k的文章的数量  
<span style='color:blue'>__由上得出每篇文章的内容特征向量：__</span> $$d_{j}=(w_{1j},w_{2j},…,w_{nj})$$  

### 用户偏好计算
<span style='color:blue'>__计算客户的文章偏好时，可直接取用户x喜欢文章的向量平均值：__</span>$$U_{x}=\frac{(d_{x1}+d_{x2}+d_{x3})}{3}=(u_{1x},u_{2x},…,u_{nx})$$

### 内容召回
<span style='color:blue'>__则用户x在文章t上的得分为：__</span>$$cos\theta=\frac{U_{x}\cdot\ d_{t}}{\lVert U_{k} \rVert\lVert d_{t} \rVert}=\frac{\sum\limits_{i=1}^n (u_{ix}\cdot\ w_{it})}{\sqrt{\sum\limits_{i=1}^n u_{ix}^2}\cdot\ \sqrt{\sum\limits_{i=1}^n w_{it}^2}}$$
>代码：


```python
import numpy as np
def cosSim(u_x,d_t):
    num1=np.dot(u_x,d_t)
    num2=np.linalg(u_x)*np.linalg(d_t)
    return num1/num2
```

### 物品排序

## 基于领域的推荐（协同算法）

### 基于用户的协同过滤算法


```python

```

### 基于物品的协同过滤算法

***核心思想：给用户推荐那些和他们之前喜欢的物品相似的物品***  
***主要利用了用户行为的集体智慧***

#### 基于共同喜欢物品的用户列表计算（购买数）:
$$w_{ij}=\frac{\left|N(i)\bigcap N(j)\right|}{\sqrt{\left|N(i)\right|*\left|N(j)\right|}}$$
$N(i)$是购买物品i的用户数  
$N(j)$是购买物品j的用户数  
PS:分母中使用了物品总购买人数做惩罚，因为热门商品经常与其他商品进行一起购买，除以总人数可以降低该商品与其他商品的相似分数
>代码：


```python
import math

def itemsim(train):
    c=dict()##物品对的购买数
    n=dict()##各个物品的购买数
    for _,items in train.items():
        for i in items.keys():
            if i not in n.keys():
                n[i]=0
            n[i]+=1
            if i not in c.keys():
                c[i]=dict()
            for j in items.keys():
                if i==j:
                    continue
                if j not in c[i].keys():
                    c[i][j]=0
                c[i][j]+=1
    w=dict()
    for i,items in c.items():
        if i not in w.keys():
            w[i]=dict()
        for j,cji in items.items():
            w[i][j]=cji/(math.pow(n[i],0.5)*math.pow(n[j],0.5))
    return w

if __name__=='__main__':
    train_data={
        'A':{'i1':1,'i2':1 ,'i4':1},
        'B':{'i1':1,'i4':1},
        'C':{'i1':1,'i2':1,'i5':1},
        'D':{'i2':1,'i3':1},
        'E':{'i3':1,'i5':1},
        'F':{'i2':1,'i4':1}
        }
    w=itemsim(train_data)
    print(w)
```

    {'i1': {'i5': 0.40824829046386296, 'i2': 0.5773502691896258, 'i4': 0.6666666666666667}, 'i3': {'i5': 0.4999999999999999, 'i2': 0.35355339059327373}, 'i5': {'i1': 0.40824829046386296, 'i3': 0.4999999999999999, 'i2': 0.35355339059327373}, 'i2': {'i1': 0.5773502691896258, 'i3': 0.35355339059327373, 'i5': 0.35355339059327373, 'i4': 0.5773502691896258}, 'i4': {'i1': 0.6666666666666667, 'i2': 0.5773502691896258}}
    

#### 基于余弦的相似度（评分数据）
<span style='color:red'>__当用户购买了却不喜欢该商品时，基于购买物品的用户列表会出现推荐错误</span>__
$$w_{ij}=\frac{\sum\limits^{len}_{k=1}(n_{ki}\cdot n_{kj})}{\sqrt{\sum\limits^{len}_{k=1}n_{ki}^2}\cdot\sqrt{\sum\limits^{len}_{k=1}n_{kj}^2}}$$
$n_{ki}$是用户k对物品i的评分
>代码：


```python
def itemcos(train):
    c=dict()
    n=dict()
    for _,items in train.items():
        for i in items.keys():
            if i not in c.keys():
                c[i]=dict()
            if i not in n.keys():
                n[i]=0
            n[i]+=items[i]*items[i]
            for j in items.keys():
                if i==j:
                    continue
                if j not in c[i].keys():
                    c[i][j]=0
                c[i][j]+=items[i]*items[j]
    w=dict()
    for i,items in c.items():
        if i not in w.keys():
            w[i]=dict()
        for j in items.keys():
            w[i][j]=c[i][j]/(math.sqrt(n[i])*math.sqrt(n[j]))
    return w

if __name__ == '__main__':  
    train = {'A':{'i1':1,'i2':1 ,'i4':1},  
     'B':{'i1':1,'i4':1},  
     'C':{'i1':1,'i2':1,'i5':1},
     'D':{'i2':1,'i3':1},
     'E':{'i3':1,'i5':1},
     'F':{'i2':1,'i4':1}
        }  
    w=itemcos(train)
    print(w)
```

    {'i3': {'i5': 0.4999999999999999, 'i2': 0.35355339059327373}, 'i1': {'i4': 0.6666666666666667, 'i5': 0.40824829046386296, 'i2': 0.5773502691896258}, 'i4': {'i1': 0.6666666666666667, 'i2': 0.5773502691896258}, 'i5': {'i1': 0.40824829046386296, 'i3': 0.4999999999999999, 'i2': 0.35355339059327373}, 'i2': {'i1': 0.5773502691896258, 'i4': 0.5773502691896258, 'i5': 0.35355339059327373, 'i3': 0.35355339059327373}}
    

#### 热门物品的惩罚
$$w_{ij}=\frac{\left|N(i)\bigcap N(j)\right|}{\left|N(i)\right|^\alpha \cdot\left|N(j)\right|^{1-\alpha}}$$
$\alpha$是物品i的惩罚系数，取（0.5,1）
>代码：


```python
def itencos_alpha(train,alpha=0.7):
    c=dict()
    n=dict()
    for _,items in trian.items():
        for i in items.keys():
            if i not in c.keys():
                c[i]=dict()
            if i not in n.keys():
                n[i]=0
            n[i]+=1
            for j in items.keys():
                if i==j:
                    continue
                if j not in c[i].keys():
                    c[i][j]=0
                c[i][j]+=1
    w=dict()
    for i.items in c.items():
        if i not in w.keys():
            w[i]=dict()
        for j,cij in items.items():
            w[i][j]=c[i][j]/(math.pow(n[i],alpha)*math.pow(n[j]))
    return w

if __name__=='__main__':
    train_data={
        'A':{'i1':1,'i2':6 ,'i4':1},
        'B':{'i1':1,'i4':1},
        'C':{'i1':1,'i2':7,'i5':1},
        'D':{'i2':8,'i3':1},
        'E':{'i3':1,'i5':1},
        'F':{'i2':9,'i4':1}
        }
    w=itemsim(train_data)
    print(w)
```

    {'i1': {'i5': 0.40824829046386296, 'i2': 0.5773502691896258, 'i4': 0.6666666666666667}, 'i3': {'i5': 0.4999999999999999, 'i2': 0.35355339059327373}, 'i5': {'i1': 0.40824829046386296, 'i3': 0.4999999999999999, 'i2': 0.35355339059327373}, 'i2': {'i1': 0.5773502691896258, 'i3': 0.35355339059327373, 'i5': 0.35355339059327373, 'i4': 0.5773502691896258}, 'i4': {'i1': 0.6666666666666667, 'i2': 0.5773502691896258}}
    

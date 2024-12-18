## day02问题小结

用户标签里面判断活跃用户还是有点不理解

```sql
#利用日期函数打上实时事实类标签
drop table if exists mall_customer_realtime_tag;
create table mall_customer_realtime_tag as 
select 
a.userid,
case when a.last_visit >= DATE_SUB(Curdate(), INTERVAL 30 DAY) then "一个月活跃" else "近一个月不活跃" end as "近一个月活跃标签",
case when a.last_visit >= DATE_SUB(curdate(),INTERVAL 90 DAY) then "近三个月活跃" else "近三个月不活跃" end as "近三个月活跃标签",
case when a.last_visit >= DATE_SUB(curdate(),INTERVAL 180 DAY) then "近六个月活跃" else "近六个月不活跃" end as "近六个月活跃标签",
case when a.last_visit >= DATE_SUB(curdate(),INTERVAL 360 DAY) then "近十二个月活跃" else "近十二个月不活跃" end as "近十二个月活跃标签"
from 
(select 
userid, max(date) as last_visit
from mall_customers_tld_header
GROUP BY userid) a
```

代码都能看懂，但是看见需求就是写不出来咋办？

- 看懂到写出来中间隔了一万行代码，多练，多思考

知道数据推断的使用场景

- 疫情期间，一家印度外卖餐厅想通过数据分析，数据挖掘提升销量，但是在历史数据中**缺少了很重要的一个维度，用餐人数**

了解用户标签的作用

- 用户标签是精细化运营的抓手
- 发现兴趣，投其所好
- 发现用户特点，提升用户价值
- 为业务发展储备标签

应用SQL实现为用户打标签

- 熟悉sql代码

知道AARRR模型的含义

- 是用户生命周期的模型，但同时也有人拿来变成运营流程的模型：先拉新，其次促活，接着提高留存，然后获取收入，最后实现自转播。
- **Acquisition（获得新用户）**：PV，UV
- **Activation（用户激活）**：用户活跃，按时间维度
- **Retention（用户留存）**：留存率（次日留存，七日留存等）
- **Revenue（用户付费）**：获取收入，用户购买率
- **Referral（用户推荐）** 分享，（朋友圈，砍一刀，返现，分享满N人给优惠券）这里缺少相关数据

竞品是什么？

竞争对手的产品乃至所有值得参照借鉴的产品 。

## 三、用户行为分析【续day02】

### 4、代码实现【掌握】

#### 4.1 数据加载与处理

加载数据

去掉无用数据

数据类型转换

查看数据基本情况，查看be_type字段总类 查看数据中是否有空值

提取出时间中的月份、天、时、星期等维度

#### 4.2 用户行为分析

根据用户行为对数据进行分组

流量指标分析：PV、UV、平均访问量

```python
PV=behavior_count['pv']
print("PV=%d"%PV)
UV=len(data1['cust_id'].unique())
print("UV=%d"%UV)
print("平均访问量 PV/UV=%d"%(PV/UV))
```

跳失率：只有点击行为的用户数/总用户数，总用户数即uv

```python
集合减法 set1-set2 得到的结果是在set1中且不在set2中的元素
pv_only=len(data_pv_only)
print('跳失率为：%.2f%%'%(pv_only/UV*100))
```

按天进行PV统计

按天进行UV统计，进行去重操作

```
uv_day=data1[data1.be_type=='pv'].drop_duplicates(['cust_id','buy_time']).groupby('buy_time')['cust_id'].count()
uv_day

```

按小时计算PV，UV

```
fig, axes = plt.subplots(2,1, figsize=(16, 12), sharex=True)  # subplots返回两个值，fig画布，axes坐标轴
pv_hour.plot(x='hours', y='pv', ax=axes[0])
uv_hour.plot(x='hours', y='uv', ax=axes[1])
plt.xticks(range(24),np.arange(24))
axes[0].set_title('按小时点击量趋势图')
axes[1].set_title('按小时独立访客数趋势图')
```

留存率，计算方法：

1 找到新用户

2 判断第n天新用户是否是活跃

3 计算留存率

```python
def cal_retention(data,n): #n为n日留存
    user=[]
    date=pd.Series(data.buy_time.unique()).sort_values()[:-n] #时间截取至最后一天的前n天
    retention_rates=[]
    for i in date:
        new_user=set(data[data.buy_time==i].cust_id.unique())-set(user) #识别新用户，本案例中设初始用户量为零
        user.extend(new_user)  #将新用户加入用户群中
        #第n天留存情况
        user_nday=data[data.buy_time==i+timedelta(n)].cust_id.unique() #第n天登录的用户情况
        a=0
        for cust_id in user_nday:
            if cust_id in new_user:
                a+=1
        retention_rate=a/len(new_user) #计算该天第n日留存率
        retention_rates.append(retention_rate) #汇总n日留存数据
    data_retention=pd.Series(retention_rates,index=date)
    return data_retention

data_retention=cal_retention(data1,3)  #求用户的3日留存情况
data_retention
```

购买人数与购买率

```
data1[data1.be_type == 'buy'].drop_duplicates(['cust_id', 'buy_time']).groupby('buy_time')['cust_id'].count() 购买用户
data1.drop_duplicates(['cust_id', 'buy_time']).groupby('buy_time')['cust_id'].count() 活跃用户
```

复购率：复购指两天以上有购买行为,一天多次购买算一次，复购率=有复购行为的用户数/有购买行为的用户总数

```
df_rebuy = data1[data1.be_type == 'buy'].drop_duplicates(['cust_id', 'day_id']).groupby('cust_id')['day_id'].count()
df_rebuy[df_rebuy >= 2].count() / df_rebuy.count()
```

购物转化漏斗分析

```
!pip install plotly
```



```
准备漏斗数据
plotly绘制漏斗图

import plotly.express as px
data = dict(
    number=[pv_users, cart_users, fav_users, buy_users],
    stage=attr)
fig = px.funnel(data, x='number', y='stage')
fig.show()
```



#### 4.3 商品维度分析

购买产品类目计数

购买产品类目前10名

商品点击计数

商品购买次数和种类统计



## 一、商品库存分析

### 学习目标

- 知道库存管理的ABC模型
- 知道库存管理的XYZ模型
- 完成ABC-XYZ建模案例

### 1、电商中的库存管理方法【了解】

#### 1.1 ABC 管理法

ABC 管理法是管理库存的经典方法。通过计算每个SKU的销售收入在所有SKU产生的总收入中的累积百分比贡献进行排名，来对商品进行分类

- A类商品贡献了80%的销售收入，这些商品需要严格控制库存，避免缺货
- B类商品贡献了接下来的10%的销售收入，对于这类商品，库存控制可以适当放松
- 剩下的10%收入由C类商品贡献，但C类商品种类最多，分配到的进货成本和仓储资源优先级最低



#### 1.2 XYZ库存管理方法

通过 XYZ 库存管理，在很长一段时间内测量每个SKU的销售需求，以捕捉需求的季节性变化，然后计算每个SKU的方差，并根据其变化对分数进行排名。

- X 类：需求变化最小的产品。 这些产品的销售比较稳定，这意味着采购经理更容易预测它们，很容易避免缺货。
- Y 类：比X类中的产品变化更大。由于季节性等因素，需求会随时间变化，因此更难准确预测
- Z 类：需求波动起伏较大，比较难预测，除非有冗余较大的备货，否则很难避免缺货

#### 1.3 将ABC 与 XYZ组合使用

对于每个SKU我们组成 AX、AY、AZ、BX、BY、BZ、CX、CY 和 CZ九个类别，通过这九个类别，我们可以帮助运营和采购人员了解两者对收入的重要性以及需求的变化

### 2、代码实现【掌握】

#### 2.1 加载数据&数据清洗

导入使用到的模块

加载交易数据

我们将日期转换成日期时间类型

调整一下列名，方便后续分析

从数据中去掉退货的商品

```
为了快速了解数据集中的季节性
sns.barplot(x="month", 
             y="quantity", 
             data=df_12m,
             palette="Blues_d")\
            .set_title("Quantity by month",fontsize=15)
```

创建销售收入字段

```
df_12m = df_12m.assign(revenue =df_12m['quantity']*df_12m['UnitPrice'])
plt.subplots(figsize=(15, 6))
sns.barplot(x="month", 
             y="revenue", 
             data=df_12m,
             palette="Blues_d")\
            .set_title("Revenue by month",fontsize=15)
```



#### 2.2 构建XYZ模型

1 分组聚合每个sku每个月的总销量

2 计算每个sku每个月销量，通过dataframe的.std(axis=1) 来计算每行中值的标准差

3 计算总需求，计算每月的平均需求

4 **计算需求变异系数**

5 计算完变异系数我们将数据按照变异系数高低排序

6 绘制 CV的直方图 直方图对连续数值，柱状图离散数值



区分XYZ

这里将0.5和1作为阈值，变异系数<0.5的作为X类，变异系数>1的作为Z类



```
df_12m_units.groupby('xyz_class').agg(
    total_skus=('sku', 'nunique'),
    total_demand=('total_demand', 'sum'),    
    std_demand=('std_demand', 'mean'),      
    avg_demand=('avg_demand', 'mean'),
    avg_cov_demand=('cov_demand', 'mean'),
)
聚合之后得出结论，容易预测的总量低
```



#### 2.3 XYZ数据可视化

对每个xyz不同类别和不同月份进行分组聚合，统计出每个月份每个类别总销量

画图展示

```
plt.subplots(figsize=(15, 6))
sns.barplot(x="month", 
             y="demand", 
             hue="xyz_class", 
             data=df_monthly_unstacked,
             palette="Blues_d")\
            .set_title("XYZ demand by month",fontsize=15)
            
hue参数设置不同类别的不同颜色
```



#### 2.4 ABC库存分析

ABC 库存分析从销售金额的角度进行分析

计算每个SKU的收入指标

对总收入进行降序排序

可以使用 cumsum() 函数计算收入的累计总和，然后计算收入的百分比并将其存储在DataFrame中。

根据计算得到的百分比对sku进行划分，

```
def abc_classify_product(percentage):
    if percentage > 0 and percentage <= 80:
        return 'A'
    elif percentage > 80 and percentage <= 90:
        return 'B'
    else:
        return 'C'
```

画图验证，A商品sku比较少，总销售金额占80%



#### 2.5 ABC-XYZ库存分析

最后一步是将 XYZ 和 ABC 标签数据结合起来。 

```
df_abc_xyz = df_abc.merge(df_xyz, on='sku', how='left')
df_abc_xyz_summary = df_abc_xyz.groupby('abc_xyz_class').agg(
    total_skus=('sku', 'nunique'),
    total_demand=('total_demand', sum),
    avg_demand=('avg_demand', 'mean'),    
    total_revenue=('total_revenue', sum),    
).reset_index()
```

### 3、业务解读【掌握】

#### ABC XYZ分类的业务解读

#### 库存管理手段应用



## 二、用户评论文本挖掘

### 学习目标

- 知道评论文本挖掘的作用
- 掌握使用nltk和gensim来进行基本NLP处理

### 1、评论文本挖掘介绍【了解】

文本挖掘定义：

- 文本挖掘就是从文本信息中挖掘我们感兴趣的内容

文本挖掘目标：

- 运营优化：挖掘用户喜好，挖掘竞品动态，提升自身产品竞争力
- 产品更新：发掘产品更新动向，及时的从用户处发现产品问题
- 口碑管理：识别出自家产品和竞争对手的口碑差异



### 2、项目背景【了解】

我们想从用户的角度了解有关竞品以及市场的信息

项目需求：

- 项目需求：
  - 竞品销售情况细化：通过对竞品评论中分型号的历史评论数量，反推竞品的主要售卖产品的情况
  - 竞品高分和低分的具体发声：高分4-5分的评论主要是说哪些，低分1-2分用户主要说什么，我们比较关注的方面又说了什么
- 技术实现：
  - 竞品细化
  - 高分、低分关键词提取

### 3、文本挖掘相关方法介绍【掌握】

#### 1、如何用数值来表示文本

向量化编码：onehot

男[1,0]

女[0,1]

北京[1,0,0]

上海[0,1,0]

深圳[0,0,1]

流程：

获取原始文本 

分词

- 首先将一句话拆分成一个一个单词，英文分词很简单，直接通过空格就可以
- 中文分词可以借助jieba这样的三方库，通过一些分词算法
- 接下来我们需要对有时态变化的单词还原成未变化的单词
- 获取原始单词之后还需要去掉停用词和一些助词，虚词，连词，没有实义的单词

向量化编码



### 4、 代码实现【掌握】

#### 1、导包&载入数据

```
# nltk：文本处理的包
from nltk.stem.wordnet import WordNetLemmatizer # 词性还原
from nltk.corpus import wordnet as wn


```

查看数据情况

df_reviews.info()

df_reviews = df_reviews.dropna() 缺失值很少，直接删掉



#### 2、数据处理

```
# 截取评论中的星级数据 
def get_stars(n):

# 根据评星数量获取评价属性， 好评（4分及以上）， 中评（3分）， 差评（2分及以下）
def stars_cat(n):

# 获取评论中的日期信息，转换成日期时间格式
def get_date(x):
```

#### 3、非文本数据的分析

- 统计产品的评论数量
- 统计不同类型的产品数量
- 统计产品评论星级分布

#### 4、文本挖掘

#### 5、创建词云图


## day04问题小结

ACU,PCU的概念要理解

- ACU（Average concurrent users）平均同时在线玩家人数，即在一定时间段抓取一次数据，以一定周期为期限；周期内的ACU可取时间段的平均数据。[例如：系统每一小时抓取一次数据，全天24小时共24个不同时刻的在线数据，则每天的ACU是这24个数据的平均值（每个公司有每个公司的定义，一般ACU取平均值，若针对某一时刻，则直接在某时刻内直接统计用户数）
- PCU（Peak concurrent users）最高同时在线玩家人数，即在一定时间内，抓取最高在线数据。（例如：单天最高在线：系统每小时统计一次数据，全天24小时共24个不同时刻的在线数据，则24个时间段内最高的用户在线数据为PCU）

datetime.strptime和datetime.strftime的用法

```
datetime.datetime.strptime?
Docstring: string, format -> new datetime parsed from a string (like time.strptime()).
Type:      builtin_function_or_method

datetime.datetime.strptime('2019-11-09 07:32:10', '%Y-%m-%d %H:%M:%S')

datetime.datetime.strftime?
Docstring: format -> strftime() style string.
Type:      method_descriptor

datetime.datetime.strftime(datetime.datetime(2019, 11, 9, 7, 32, 10), '%Y-%m-%d')
```

知道游戏数据分析的常用方法

- 运营数据
  - 激活数据

  - **激活且登录率应用场景**
  - 活跃数据
  - 在线数据

- 游戏行为数据

- 市场投放数据
- 用户付费指标
- 转化率漏斗

...



## RFM用户价值分析

### 学习目标

- 掌握RFM的基本原理
- 利用RFM模型从产品市场财务营销等不同方面对业务进行分析

### 1、RFM基本原理【掌握】

- 最近一次消费（Recency）
- 消费频率（Frequency）
- 消费金额（Monetary）

统计用户的这三个指标，分别对这三个指标进行划分，每个指标可以分成5份/3份，得到不同客群，有针对性进行营销推广



### 2、RFM实战——Pandas数据处理【掌握】

#### 2.1 加载数据与基本数据处理

加载数据

去掉缺失值所在的行

```
orders.dropna(inplace = True)
```

替换数据中的英文

```
orders['product'] = orders['product'].replace('banana', '香蕉')
# 另外一种替换方式
orders['product'].replace('milk', '牛奶',inplace = True)
orders['product'].replace('water', '水',inplace = True)
```

#### 2.2 RFM计算

计算每个用户的购物情况

频率计算

最近一次消费计算

合并recency、frequency

划分recency、frequency 

- pd.cut进行划分，把指定的列按照指定的范围进行划分，可以给每个范围一个标签

#### 2.3 RFM分析

```
# RF交叉分析
RF_table = pd.crosstab(purchase_list['frequency_cate'].astype(str),
                       purchase_list['recency_cate'].astype(str))
                       
根据RF标注出顾客类别 常客,新客,沉睡客,流失客
频次>=4 & 最近一次时间<=22 常客
频次>=4 & 最近一次时间>22 沉睡客
频次<4 & 最近一次时间>22 流失客
剩下的是新客
```



### 3、RFM可视化【掌握】

#### 3.1 Seaborn绘图

sns.FacetGrid



#### 3.2 matplotlib绘制RFM

利用matplotlib 搭框架, Seaborn绘制里面的小图



### 4、RFM分析-产品分析【掌握】

#### 4.1 RFM图调整 XY轴统一标签

 XY轴统一标签

设置axes设定xlabel ylabel

#### 4.2 统一Y轴刻度

找到某个产品最大值，根据该值进行设置



#### 4.3 四大类顾客分群

区分出, 常客, 新客, 沉睡客, 流失客 我们给四个区块添加背景颜色



#### 4.4 RFM产品分析

业务解读:

- 从图中可以很直观的发现不同类型的产品在不同客群中的消费情况
- 比如水在各群组中销售的情况都比较好, 牛奶在常客群中销售情况较好

### 5、RFM分析-市场分析【掌握】

#### 5.1 市场分析

我们依然将客户按R和F划分成4个组, 每个柱状图显示一个性别, 用累计柱状图来显示不同产品的购买比例

#### 5.2 市场分析图例优化

每个小图都有一个图例， 而且每个图例的内容都是一样的， 我们可以让图例只出现一次

- 绘制每个小图时指定参数legend = False, 在小图中关闭图例
- 并且在绘图过程中,绘制一次图例即可

### 6、RFM分析 - 财务分析【掌握】

#### 6.1 成本获利分析

- 需要注意, 成本数据可能需要估算, 不见的每家公司都会有精确的用户成本数据
- 我们可以在做运营活动的时候为每一个涉及到的会员打上标签, 分摊运营成本, 从而估算出每个用户的成本
- Customer Acquisition Cost (CAC)

#### 6.2 RFM毛利率分析

为了进一步分析清楚, 那些用户身上是赚钱的, 哪些用户是赔钱的,我们可以计算毛利

#### 6.3 RFM 投资回报率分析

计算ROI投资回报率

- 投入一元钱成本会带来多少收益
- 不同行业ROI不同,我们分析时设定的阈值可以根据自身行业做调整



### 7、RFM营销分析【掌握】

- 在上一小节中,我们通过ROI分析, 得出某些客群的ROI较低, 接下来我们要尝试把这部分低ROI的用户的营销预算做一些调整,比如减少一半
- 我们把在低ROI客群的营销费用转移到高ROI的客群,我们来试算一下, 经过这样的调整,我们的ROI是否会有提升
- 这里假设不同客群的ROI在试算过程中是固定的

### 8、RFM顾客复购分析【掌握】

#### 8.1 复购分析

评估下面三个用户的活跃程度，如果只是考虑次数， 不考虑间隔， 这三个用户活跃程度一样

考虑加权计算活跃

$\large 1-\frac{\sum_i(第i次复购时间间隔*i)}{平均复购间隔*\sum_i复购次数}$

#### 
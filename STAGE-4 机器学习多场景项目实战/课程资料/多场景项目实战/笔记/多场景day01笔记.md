## 一 指标计算

#### 学习目标

- 掌握数据指标的概念
- 知道常见的业务指标含义
- 掌握常用数据指标计算方法

### 1、数据指标简介

#### 1.1 什么是数据指标【了解】

- 数据指标概念：可将某个事件量化，且可形成数字，来衡量目标。
- 数据指标的作用：当我们确定下来一套指标，就可以用指标来衡量业务，判断业务好坏

#### 1.2 常用的业务指标【掌握】

- **活跃用户指标**：一个产品是否成功，如果只看一个指标，那么这个指标一定是活跃用户数
  - 日活（DAU）：一天内日均活跃设备数
  - 月活（MAU）：一个月内的活跃设备数
  - 周活跃数（WAU）：一周内活跃设备数
  - 活跃度（DAU/MAU）：体现用户的总体粘度，衡量期间内每日活跃用户的交叉重合情况
- **新增用户指标**：主要是衡量营销推广渠道效果的最基础指标
  - 日新增注册用户量：统计一天内，即指安装应用后，注册APP的用户数。
  - 周新增注册用户量：统计一周内，即指安装应用后，注册APP的用户数。
  - 月新增注册用户量：统计一月内，即指安装应用后，注册APP的用户数。
  - 注册转化率：从点击广告/下载应用到注册用户的转化。
  - DNU占比：新增用户占活跃用户的比例，可以用来衡量产品健康度
    - 新用户占比活跃用户过高，那说明该APP的活跃是靠推广得来
- **留存指标**：是验证APP对用户吸引力的重要指标。通常可以利用用户留存率与竞品进行对比，衡量APP对用户的吸引力
  - 次日留存率：某一统计时段新增用户在第二天再次启动应用的比例
  - 7日留存率：某一统计时段新增用户数在第7天再次启动该应用的比例，14日和30日留存率以此类推
- **行为指标**：
  - PV（访问次数，Page View）：一定时间内某个页面的浏览次数，用户每打开一个网页可以看作一个PV。
  - UV（访问人数，Unique Visitor）：一定时间内访问某个页面的人数。
  - 转化率：计算方法与具体业务场景有关
    - 淘宝店铺，转化率=购买产品的人数／所有到达店铺的人数
    - 在广告业务中，广告转化率=点击广告进入推广网站的人数／看到广告的人数。
  - 转发率：转发率=转发某功能的用户数／看到该功能的用户数
- **产品数据指标**
  - GMV （Gross Merchandise Volume）：指成交总额，也就是零售业说的“流水”
  - 人均付费=总收入／总用户数
    - 人均付费在游戏行业叫**ARPU**（Average Revenue Per User）
    - 电商行业叫**客单价**
  - 付费用户人均付费（ARPPU，Average Revenue Per Paying User）=总收入／付费人数，这个指标用于统计付费用户的平均收入
  - 付费率=付费人数／总用户数。付费率能反映产品的变现能力和用户质量
  - 复购率是指重复购买频率，用于反映用户的付费频率。
- **推广付费指标**：
  - CPM（Cost Per Mille） ：展现成本，或者叫千人展现成本
  
  - CPC（Cost Per Click） 点击成本，即每产生一次点击所花费的成本
  
  - 按投放的实际效果付费（CPA，Cost Per Action）包括：
  
    - CPD（Cost Per Download）：按App的下载数付费；
    - CPI（Cost Per Install）：按安装App的数量付费，也就是下载后有多少人安装了App；
    - CPS（Cost Per Sales）：按完成购买的用户数或者销售额来付费。
  
- 不同的业务可能关心的指标不尽相同

#### 1.3 如何选择指标【了解】

- 好的数据指标应该是比例
- 根据目前的业务重点，找到北极星指标
  - 在实际业务中，北极星指标一旦确定，可以像天空中的北极星一样，指引着全公司向着同一个方向努力

### 2、Python指标计算案例【掌握】

数据中包含了某电商网站从2009年12月到2011年12月两年间的销售流水, 每条记录代表了一条交易记录, 包含如下字段

`Invoice`: 发票号码

`StockCode`: 商品编码

`Description`: 商品简介

`InvoiceDate`: 发票日期

`Price`: 商品单价

`Customer ID`: 用户ID

`Country`: 用户所在国家

计算的指标

- 月销售金额(月GMV) 

- 月销售额环比

- 月销量

- 新用户占比: 新老用户
- 激活率
- 月留存率

#### 2.1 导入模块&加载数据

```
from datetime import datetime, timedelta
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data_1 = pd.read_excel('data/online_retail_II.xlsx',sheet_name='Year 2009-2010')
data_2 = pd.read_excel('data/online_retail_II.xlsx',sheet_name='Year 2010-2011')
```

#### 2.2 数据清洗

```
retail_data['购买时间'].describe()
retail_data.describe()

retail_data_clean = retail_data[(retail_data['商品单价']>0) & (retail_data['购买数量']>0)]
```



#### 2.3 计算月销量指标

 `商品编号`相当于 SKU

- SKU=Stock Keeping Unit（库存量单位）

```
retail_data_clean = retail_data_clean.query("(商品编号!='B') and (商品编号!='TEST001') and (商品编号!='TEST002') ")

去掉某些商品编号

retail_data_clean['购买年月'] = retail_data_clean['购买时间'].astype('datetime64[M]')

retail_data_clean['金额'] = retail_data_clean['商品单价'] * retail_data_clean['购买数量']

gmv_m = retail_data_clean.groupby(['购买年月'])['金额'].sum().reset_index()

gmv_m.columns = ['购买年月', '月GMV']

画图显示
```



#### 2.4 计算月销售额环比

环比概念: 当前月跟上一月对比

```
gmv_m['金额'].pct_change()
pd.Series.pct_change? 直接在jupyter 单元格中运行可以查看文档 该函数计算当前单元格和上一个单元格差异的百分比

可视化月销售额环比数据

1月数据下降可以理解，但是4月份环比数据也有明显下降，我们来进一步分析
```



#### 2.5 月均活跃用户分析

数据中只有购买记录，没有其它记录，所以我们**用购买行为来定义活跃**

```
mau = retail_data_clean.groupby('购买年月')['用户ID'].nunique().reset_index()
统计每个月去重之后用户数量
```

merge

how : left right outer inner

on 两表都有的列

left_on 左表列

right_on 右表列



#### 2.6 月客单价(活跃用户平均消费金额)

客单价 = 月GMV/月活跃用户数

```
final['客单价'] = final['金额']/final['用户数']
```

4月份活跃用户 和客单价都在下降，所以月GMV下降



#### 2.7 新用户占比

根据用户最近一次购买和第一次购买时间的差异，如果相同，则认为是新用户，否则老用户

```
retail_data_clean.groupby(['购买年月','用户类型'])['金额'].sum().reset_index()
分组统计后得到新老用户购买金额
里我们需要把新老用户的购买情况都绘制到一张图表中
```



新用户占比= 每月新用户/每月有购买的总用户数



#### 2.8 激活率计算

- 用户激活的概念：用户激活不等同于用户注册了账号/登录了APP，不同类型产品的用户激活定义各有差别
- 总体来说，用户激活是指用户一定时间内在产品中完成一定次数的关键行为

```
# 统计每月激活用户数量
activation_count = retail[retail['首次购买年月'] == retail['注册年月']].groupby('注册年月')['用户ID'].count()
# 统计每月注册的用户数
regist_count = retail.groupby('注册年月')['用户ID'].count()


```

```
#按渠道统计每月不同渠道的激活用户数
activation_count = retail[retail['首次购买年月'] == retail['注册年月']].groupby(['注册年月','渠道'])['用户ID'].count()
#按渠道统计每月注册用户数
regist_count = retail.groupby(['注册年月','渠道'])['用户ID'].count()
#计算不同渠道激活率
```



#### 2.9 月留存率

月留存率 = 当月与上月都有购买的用户数/上月购买的用户数

```
user_purchase.pivot_table
index = '用户ID',columns= '购买年月',values='购买数量', aggfunc=sum

for循环计算每个月留存
```


## RFM用户价值分析

<img src="img/rfm_1.png" style="zoom:150%;" />

### 学习目标

- 掌握RFM的基本原理
- 利用RFM模型从产品市场财务营销等不同方面对业务进行分析



### 1、RFM基本原理

- 在各种数据分析模型中， 有一项工具可以帮助公司找到

  - R 新客（近期有消费的用户）
  - F 常客（常来消费的用户）
  - M 贵客（消费金额大的用户）

- 这个工具就是RFM模型， RFM模型是由（George Cullinan）于1961年提出，他发现数据分析中，有三项重要的指标：

  - 最近一次消费（Recency）
  - 消费频率（Frequency）
  - 消费金额（Monetary）
  - 这三项指标的英文首字母为R、F、M，所以就称为「RFM模型」

  <img src="./img/rfm_2.png" align='left' />

  接下来,介绍一下RFM模型的基本用法

  1. 最近一次消费（Recency）：指用户上次购买时刻到现在的时间差, **例如**我们将购买日距今时间划分为5等分:

  - 最近消费的前20%，R分值为5
  - 20%~40%R分值为4, 以此类推
  - 最后20%的用户, R分值为1

  > R分值越低, 说明越久没有来,流失可能性较高, R分值高的,活跃度比较好

  2. 消费频率（Frequency）：指用户在一定时间内的消费次数，例如：

  - 次数最多的前20%，F分值为5
  - 20%~40%F分值为4，以此类推
  - 消费次数最少的最后20%， F分值为1

  > **F分值越高的用户，其消费频率越高，忠诚度与LTV越高**。

  3. 消费金额（Monetary）：指消费者在**一定时间内购买商品的总金额**， 例如：

  - 金额最大的前20%，M分值为5
  - 20%~40%M分值为4，以此类推
  - 消费金额最少的最后20%， M分值为1

  > **M分值越高的消费者，其消费金额越高，顾客价值也越高。**

- 利用上面的打分方式， 我们可以将顾客根据（R,F,M）的分数分成125群， 从最低的（1,1,1）（3分）到最高的（5,5,5）（15分）
- RFM模型能协助企业区分顾客, 进而分析不同客群背后的消费行为习惯, 是我们的运营活动更加有的放矢

### 2、RFM实战——Pandas数据处理

- 在下面的实战中, 我们将使用RFM模型, 对数据进行处理, 并对RF交叉数据进行可视化, 结合可视化图形,我们分别从产品、市场、财务、营销、复购等5个方面进行深入分析，解释业务运行状态，并给出相应建议

![png](img/output_59_1.png)

![png](img/output_76_1.png)

![png](img/output_92_1.png)

#### 2.1 加载数据与基本数据处理


```python
import pandas as pd
import datetime
import numpy as np
```


```python
orders= pd.read_csv('data/orders.csv',index_col=0)
orders
```

<img src="img/rfm_3.png"  align='left'/>

- grossmarg 毛利率


```python
orders.info()
```

<img src="img/rfm_4.png"  align='left'/>


- 这里为了演示RFM的划分, 缺失数据直接删除


```python
orders.dropna(inplace = True)
```

- 替换数据中的英文


```python
orders['product'] = orders['product'].replace('banana', '香蕉')
# 另外一种替换方式
orders['product'].replace('milk', '牛奶',inplace = True)
orders['product'].replace('water', '水',inplace = True)
```

#### 2.2 RFM计算

- 计算每个用户的购物情况


```python
orders['values'] = 1
purchase_list = orders.pivot_table(index=['clientId','orderId','gender','orderdate'], #行索引
                          columns='product', # 列索引
                          aggfunc=sum, # 计算方式，max, min, mean, sum, len
                          values='values' #值
                          ).fillna(0).reset_index()
purchase_list
```

<img src="img/rfm_5.png"  align='left'/>

- 这里使用3-4个商品比较合适, 热销商品,主力商品, 新品, 或者想对比的商品

- 频率计算
    - 计算每个用户在一定时间内购买该产品的次数


```python
# 创建一列 frequency 辅助计算
purchase_list['frequency'] = 1
frequency = purchase_list.groupby("clientId", #按照用户ID进行分组
                                  as_index = False # 分类的列是否做为行索引
                                  )['frequency'].sum() # 聚合方式，max, min, mean, sum
frequency
```

<img src="img/rfm_6.png"  align='left'/>


```python
# 完成计算后可以将辅助计算列删除
del purchase_list['frequency']
purchase_list
```

<img src="img/rfm_7.png"  align='left'/>

- 最近一次消费计算


```python
# 将数据中最后一天作为观察日期, 以这一天为基准分析RFM情况
theToday = datetime.datetime.strptime(orders['orderdate'].max(), "%Y-%m-%d")
# 将数据中的'orderdate'列处理成日期时间格式
purchase_list['orderdate'] = pd.to_datetime(purchase_list['orderdate'])
# 计算每个顾客最近一次购买日期
recent_recency = purchase_list.groupby("clientId", as_index = False)['orderdate'].max()
# 计算 观察日 与 每位用户最后一次购买日期的时间差
recent_recency['recency'] =( theToday - recent_recency['orderdate'] ).astype(str)
recent_recency
```

<img src="img/rfm_8.png"  align='left'/>




```python
# 去掉recency列中的days
recent_recency['recency'] = recent_recency['recency'].str.replace('days.*', #要替换的内容
                                                                  '', #替换成的字符串
                                                                  regex = True)
# 'recency'列转换成Int
recent_recency['recency'] = recent_recency['recency'].astype(int)
recent_recency
```

<img src="img/rfm_9.png"  align='left'/>



- 合并recency、frequency


```python
purchase_list = recent_recency.merge(purchase_list, # 要合并的DataFrame
                                     on = ['clientId', 'orderdate'] # 链接的Key
                                     ,how='inner') # 合并的方式
```


```python
purchase_list =purchase_list.merge(frequency, # 要合并的DataFrame
                                   on = ['clientId'] # 链接的Key
                                   ,how='inner') # 合并的方式
purchase_list
```

<img src="img/rfm_10.png"  align='left'/>

- 划分recency、frequency


```python
# 将recency按照时间远近分组
recency_label =  ['0-7 day', '8-15 day', '16-22 day', '23-30 day', '31-55 day', '>55 day']
# cut 自定义的方式对数据进行分组, 默认 左开右闭
recency_cut  = [-1, 7, 15, 22, 30, 55, purchase_list['recency'].max()]
purchase_list['recency_cate'] = pd.cut( 
        purchase_list['recency'] , #要分组的列
        recency_cut, #划分条件
        labels =recency_label) #每组的名字

# 将frequency按照频率高低分组
frequency_label =  ['1 freq', '2 freq', '3 freq', '4 freq', '5 freq', '>5 freq']
frequency_cut  = [0, 1, 2, 3, 4, 5, purchase_list['frequency'].max()]
purchase_list['frequency_cate'] = pd.cut( 
        purchase_list['frequency'] , #要分组的列
        frequency_cut,  #划分条件
        labels =frequency_label) #每组的名字
purchase_list
```

<img src="img/rfm_11.png"  align='left'/>

#### 2.3 RFM分析


```python
# RF交叉分析
RF_table = pd.crosstab(purchase_list['frequency_cate'].astype(str),
                       purchase_list['recency_cate'].astype(str))

# 重新排序
RF_table['freq'] = RF_table.index
RF_table = RF_table.sort_values('freq',ascending = False)

collist = ['freq'] + recency_label
RF_table = RF_table[collist]
```


```python
RF_table
```

<img src="img/rfm_12.png"  align='left'/>


```python
# 根据RF标注出顾客类别 常客,新客,沉睡客,流失客
purchase_list['customer'] = np.where( (purchase_list['frequency'] >=frequency_cut[4]) & (purchase_list['recency']<=recency_cut[3]), '常客',
                     np.where( (purchase_list['frequency'] >=frequency_cut[4]) & ( purchase_list['recency']>recency_cut[3]), '沉睡客',
                              np.where( (purchase_list['frequency'] < frequency_cut[4]) & ( purchase_list['recency']>recency_cut[3]), '流失客',
                                       '新客'  )))

purchase_list.to_csv('purchase_list.csv')
purchase_list
```

<img src="img/rfm_13.png"  align='left'/>

### 3、RFM可视化

#### 3.1 Seaborn绘图


```python
import matplotlib.pyplot as plt
import seaborn as sns
rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False,
     'figure.figsize':(10,8)}
sns.set(context='notebook', style='ticks', rc=rc)
```


```python
#准备XY轴标签
recency_label =  ['0-7 day', '8-15 day', '16-22 day', '23-30 day', '31-55 day', '>55 day']
frequency_label =  ['1 freq', '2 freq', '3 freq', '4 freq', '5 freq', '>5 freq']

#绘图
g = sns.FacetGrid(purchase_list, # 数据源
                  col="recency_cate", # X轴数据来源
                  row="frequency_cate" ,  # Y轴数据来源
                  col_order= recency_label,  # X方向数据顺序
                  row_order= frequency_label, # Y方向数据顺序
                  palette='Set1',  #画布色调
                  margin_titles=True)
#小图表部分
g = g.map_dataframe(sns.barplot, y ='水')
g = g.set_axis_labels('距今','频率').add_legend()
```


  ![png](img/output_40_0.png)
    


#### 3.2 matplotlib绘制RFM

- 后续的RFM图形绘制 我们使用MatPlotLib与Seaborn配合
    - 利用matplotlib 搭框架, Seaborn绘制里面的小图
    - 每个小图代表一个RF交叉的组合, 如左上角我们要绘制的是: 来过店里5次以上, 并且最近一周内来过店里的情况


```python
#准备XY轴标签
recency_label =  ['0-7 day', '8-15 day', '16-22 day', '23-30 day', '31-55 day', '>55 day']
frequency_label =  ['1 freq', '2 freq', '3 freq', '4 freq', '5 freq', '>5 freq']


#绘图
#先设定画布大小   6*6 代表有36张小图
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 最近几天内来过
        data = purchase_list[(purchase_list['recency_cate']==j) & (purchase_list['frequency_cate']==i)]
        if data.shape[0] != 0: # 检查这个位置有没有数据
            # 以下为单一小图表的设定
            sns.barplot(y="牛奶", # 小图表Y数据
                        data=data, #数据来源
                        estimator=np.sum,
                        capsize=.2, # 最高点最低点大小
                        ax=axes[countX, countY]) # 小图表坐标

        countY += 1 
    countX += 1 
fig.suptitle('RFM图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # 设定X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # 设定Y轴标题
fig.show()
```




![png](img/output_43_1.png)
    


### 4、RFM分析-产品分析

#### 4.1 RFM图调整 XY轴统一标签

- 接下来我们进一步将RFM图细化, 首先把X轴 Y轴的标签填补上


```python
#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = purchase_list[(purchase_list['recency_cate']==j) & (purchase_list['frequency_cate']==i)]
        if data.shape[0] != 0: #检查这部分有没有数据
            # 下面设定单一小图表
            sns.barplot(y="牛奶", # 小图标Y数据来源
                        data=data, #绘图使用数据
                        estimator=np.sum,
                        capsize=.2, # 最高点最低点大小
                        ax=axes[countX, countY]) # 小图表坐标
        ################ 画X标记################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ############### 画Y标记 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
        countY += 1 
    countX += 1 
fig.suptitle('RFM图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```




![png](img/output_46_1.png)
    


#### 4.2 统一Y轴刻度

- 目前还有一个问题, 就是不同的小图 Y轴数值大小不同, 不方便比较, 我们要将所有的小图Y轴刻度大小


```python
findbig=0
for i in frequency_label:
    for j in recency_label:
        data = purchase_list[(purchase_list['recency_cate']==j) & (purchase_list['frequency_cate']==i)]
        if data['牛奶'].sum() > findbig:
            findbig = data['牛奶'].sum()


#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 由于axes画布排列的关系，频率必须要反着放
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = purchase_list[(purchase_list['recency_cate']==j) & (purchase_list['frequency_cate']==i)]
        if data.shape[0] != 0: #检查这部分有没有数据
            # 下面设定单一小图表
            sns.barplot(y="牛奶", # 小图标Y数据来源
                        data=data, #绘图使用数据
                        estimator=np.sum,
                        capsize=.2, # 最高点最低点大小
                        ax=axes[countX, countY]) # 小图表坐标

        ################ 将水牛奶香蕉的字体变大 ################
        axes[countX][countY].tick_params(labelsize=15)
        ############### 使所有数据尺码相同 ################
        axes[countX][countY].set_yticks(range(0,int(findbig*1.3),10))
            
        countY += 1 
    countX += 1 
fig.suptitle('RFM图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()

```






![png](img/output_49_1.png)
    


#### 4.3 四大类顾客分群

- 为了更好的区分出, 常客, 新客, 沉睡客, 流失客 我们给四个区块添加背景颜色
- 这里使用小图的坐标 (一共是 6*6 = 36张图  我们的绘图代码中用 countX  countY表示)
    - countX  0,1,2 为一组  3,4,5 为一组 countY同理 我们把整个区域划分成4各部分


```python
findbig=0
for i in frequency_label:
    for j in recency_label:
        data = purchase_list[(purchase_list['recency_cate']==j) & (purchase_list['frequency_cate']==i)]
        if data['牛奶'].sum() > findbig:
            findbig = data['牛奶'].sum()


#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 由于axes画布排列的关系，频率必须要反着放
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = purchase_list[(purchase_list['recency_cate']==j) & (purchase_list['frequency_cate']==i)]
        if data.shape[0] != 0: #检查这部分有没有数据
            # 下面设定单一小图表
            sns.barplot(y="牛奶", # 小图标Y数据来源
                        data=data, #绘图使用数据
                        estimator=np.sum,
                        capsize=.2, # 最高点最低点大小
                        ax=axes[countX, countY]) # 小图表坐标
############### 四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #紅色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黄色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1 
fig.suptitle('RFM图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```






![png](img/output_52_1.png)
    


- 需要注意的是, 这里把顾客划分成4个群体是为了演示方便, 实际业务中,我们可以根据业务情况进一步划分成更多的群体

#### 4.4 RFM产品分析

- 准备数据绘制完整的RFM分析图, 先去掉部分数据


```python
purchase_list.drop(columns = ['orderId','orderdate','recency','frequency'])
```

<img src="img/rfm_14.png"  align='left'/>


```python
temp = purchase_list.drop(columns = ['orderId','orderdate','recency','frequency'])v
# 将部分数据 宽变长, id_vars 保留不去处理的字段   
df3 = pd.melt(temp, id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values') 
df3
```

<img src="img/rfm_15.png"  align='left'/>


```python
df3 = pd.melt(purchase_list.drop(columns = ['orderId','orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values') 
df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
df3 = df3.dropna()

#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 最近R标签数据
        data = df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)]
        if data.shape[0] != 0: #检查这部分有没有数据
            # 下面绘制每一个小图表
            sns.barplot(x="types", # 小图表X轴数据
                        y="values", # 小图表Y轴数据
                        data=data, #绘图用到的数据
                        capsize=.2, 
                        ax=axes[countX, countY]) # 小图表坐标
        # ################ 画X轴刻度 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ############### 画Y轴刻度 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
            
        ################ 将水、牛奶、香蕉的字变大################
        axes[countX][countY].tick_params(labelsize=15)
        ############### 统一所有小图的Y轴刻度大小 ################
        axes[countX][countY].set_yticks(range(0,10,3))
        
        
        ###############四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黃色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1 

fig.suptitle('RFM产品分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```




![png](img/output_59_1.png)
    


- 业务解读:
    - 从图中可以很直观的发现不同类型的产品在不同客群中的消费情况
    - 比如水在各群组中销售的情况都比较好, 牛奶在常客群中销售情况较好
    
- 在产品分析中, 我们可以把主打产品, 或者是要推的新产品放在图中分析

### 5、RFM分析-市场分析

#### 5.1 市场分析

- 接下来我们可以从不同的维度进行拆解, 挖掘不同产品的市场需求情况
    - 我们依然将客户按R和F划分成4个组, 每个柱状图显示一个性别, 用累计柱状图来显示不同产品的购买比例


```python
df3 = pd.melt(purchase_list.drop(columns = ['orderId','orderdate','recency','frequency']), id_vars=['clientId','customer','recency_cate','frequency_cate','gender'], var_name='types', value_name='values') 
df3['values'] = pd.to_numeric(df3['values'],errors='coerce')
df3 = df3.dropna()

#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)]
        if data.shape[0] != 0: # 检查这部分有没有数据
            #处理堆叠数据，将这部分数据的购买量,换算成百分比data = data.groupby(['types', 'gender'])['values'].sum() # 依照不同的性別，对购买量求和
            data = data[['gender','types','values']].groupby(['types','gender']).sum()
            data =data.groupby(level=1).apply(lambda x:100 * x / float(x.sum())) # 换算成百分比
            data = data.add_suffix('').reset_index() #multiIndex 变平
            data=data.pivot('gender', 'types', 'values') # 透视表
            
           # 下面设定单一小图表
            ax = data.plot.bar(stacked=True, # 设置堆积图
                              width=0.7,# 柱子的宽度
                              legend = True, 
                              ax =axes[countX, countY] , # 小图表坐标
                              rot=0) #坐标轴文字旋转
            
        ################ 画X标记 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ############### 画Y标记 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
            
        ################ 将水、牛奶、香蕉的字变大 ################
        axes[countX][countY].tick_params(labelsize=15)
        
        
        ############### 四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黃色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1
fig.suptitle('RFM-市场分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```


![png](img/output_64_1.png)
    


- 查看数据处理的情况


```python
recency_label =  ['0-7 day', '8-15 day', '16-22 day', '23-30 day', '31-55 day', '>55 day']
frequency_label =  ['1 freq', '2 freq', '3 freq', '4 freq', '5 freq', '>5 freq']
```


```python
data = df3[(df3['recency_cate']=='0-7 day') & (df3['frequency_cate']== '>5 freq')]
```


```python
data.head()
```

[//]: # ( ![image-20210730013023157]&#40;C:\Users\Administrator\Downloads\RFM\img/rfm_16.png&#41;)
 ![image-20210730013023157](img/rfm_16.png)


```python
data = data[['gender','types','values']].groupby(['types','gender']).sum()
data
```

<img src="img/rfm_17.png"  align='left'/>

- 这里对两列分组聚合, 得到的是MultiIndex


```python
data.index.levels[1]
```


    Index(['女性', '男性'], dtype='object', name='gender')

- 此时可以通过level操作MultiIndex的数据, 也可以将其变为普通索引


```python
# 参数level 如果是MultiIndex 可以按照level 的索引进行分组
data =data.groupby(level=1).apply(lambda x:100 * x / float(x.sum())) # 将具体值换成百分比表示
data
```

<img src="img/rfm_19.png"  align='left'/>


```python
data = data.add_suffix('').reset_index() #将MultiIndex转变成普通索引
data
```

<img src="img/rfm_18.png"  align='left'/>

- 创建透视表，完成数据准备


```python
data=data.pivot('gender', 'types', 'values') # 透视
data
```

<img src="img/rfm_20.png"  align='left'/>


```python
ax = data.plot.bar(stacked=True, # 设定堆积图
                              width=0.7,# 柱子的宽度
                              legend = True, 
                              rot=0) 
```


![png](img/output_74_0.png)
    

#### 5.2 市场分析图例优化

- 上面的图中，每个小图都有一个图例， 而且每个图例的内容都是一样的， 我们可以让图例只出现一次
  - 绘制每个小图时指定参数legend = False, 在小图中关闭图例
  - 并且在绘图过程中,绘制一次图例即可


```python
#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = df3[(df3['recency_cate']==j) & (df3['frequency_cate']==i)]
        if data.shape[0] != 0: #检查这部分有没有数据
            # 处理堆叠数据，将这部分数据的购买量,换算成百分比data = data.groupby(['types', 'gender'])['values'].sum() # 依照不同的性別，对购买量求和
            data = data[['gender','types','values']].groupby(['types','gender']).sum()
            data =data.groupby(level=1).apply(lambda x:100 * x / float(x.sum())) # 换算成百分比
            data = data.add_suffix('').reset_index() #multiIndex 变平
            data=data.pivot('gender', 'types', 'values') # 透视表
            
             # 下面设定单一小图表
            ax = data.plot.bar(stacked=True, # 设置堆积图
                              width=0.7,# 柱子的宽度
                              legend = False, 
                              ax =axes[countX, countY] , # 小图表坐标
                              rot=0) #坐标轴文字旋转
            
        ################ 设定图例 ################
        if (i == '4 freq') and (j == '>55 day'):
            ax.legend(bbox_to_anchor=(1.03, 0.8), loc=2, fontsize =20) #设定图例
            
        ################ 画X标记 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ###############  画Y标记 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
            
        ################ 将水、牛奶、香蕉的字变大 ################
        axes[countX][countY].tick_params(labelsize=15)
        
        
        ############### 四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黃色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1 
fig.suptitle('RFM-市场分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()

```




![png](img/output_76_1.png)
    


- 业务解读：从上图中，我们可以对比出， 不同客群中， 不同性别的消费习惯， 从而有针对性的设计营销活动
- 我们在分析的时候， 也可以结合之前的产品分析图做对比， 更直观的讲产品的销售情况按照不同的维度进行拆解

### 6、RFM分析 - 财务分析

- 这一小节中，我们从成本和利润的角度去对不同客群进行财务方面的分析
    - 从结果中可以看出, 我们的流失用户的利润率较低, 常客的利润率较高, 原因也比较容易理解
        - 我们每一次的营销活动, 流失会员都会分摊成本, 但是没有新的收入

- 当企业处于不同的成长阶段, 分析的结果也会有不同
    - 如果是公司处于创业阶段, 业务初期,主要的营收会来自于新顾客
    - 如果是一个经营了几年的公司, 我们的利润大概率会来自于常客,而不是新客,
    - 如果一个经营了几年的公司利润来源发生了转移,新客带来的利润较多, 我们需要进一步挖掘原因, 并通过运营手段把新用户变成常客

#### 6.1 成本获利分析

- 加载成本数据
    - 需要注意, 成本数据可能需要估算, 不见的每家公司都会有精确的用户成本数据
    - 我们可以在做运营活动的时候为每一个涉及到的会员打上标签, 分摊运营成本, 从而估算出每个用户的成本
    - Customer Acquisition Cost (CAC)


```python
cac = pd.read_csv('data/cac.csv')
cac
```

<img src="img/rfm_21.png"  align='left'/>


```python
cac = cac[['clientId', 'cac']] #去掉多余数据
cac
```

<img src="img/rfm_22.png"  align='left'/>


```python
# 计算clvs
clv = orders[['clientId','grossmarg']].groupby('clientId').sum().reset_index() #计算每个顾客的总消费金额
clv
```

<img src="img/rfm_23.png"  align='left'/>


```python
clv.columns = ['clientId', 'clv'] #重命名列名
clv
```

<img src="img/rfm_36.png"  align='left'/>


```python
# purchase_list,clv,cac 合起来（merge）
purchase_list =  purchase_list.merge(clv,on=['clientId'])
purchase_list =  purchase_list.merge(cac,on=['clientId'])
purchase_list
```

<img src="img/rfm_37.png"  align='left'/>


```python
# 计算不同分组的clv与cac
countfinal = purchase_list[['frequency_cate', 'recency_cate','clv', 'cac']].groupby(['frequency_cate', 'recency_cate']).sum().reset_index()
countfinal
```

<img src="img/rfm_38.png"  align='left'/>


```python
# 将clv与cac做melt转换
finaldf = pd.melt(countfinal, id_vars=['frequency_cate','recency_cate'], value_vars=['clv', 'cac'])
finaldf
```

<img src="img/rfm_39.png"  align='left'/>


```python
# 缺失值补0
finaldf['value'] = finaldf['value'].fillna(0)
finaldf.to_csv('finaldf.csv' , index=0)
```

- 绘图


```python
#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 #  画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = finaldf[(finaldf['recency_cate']==j) & (finaldf['frequency_cate']==i)]
        if data.shape[0] != 0: #  检查这部分有没有数据
            # 下面设定单一小图表
            sns.barplot(x="variable", # 指定小图标X轴数据
                        y="value", # 指定小图标y轴数据
                        data=data, #绘图使用的DataFrame
                        capsize=.2, 
                        ax=axes[countX, countY]) # 小图表坐标
            
            
        ################ 设定图例 ################
        # if (i == '4 freq') and (j == '>55 day'):
        #     axes.legend(bbox_to_anchor=(1.03, 0.8), loc=2, fontsize =20) #设定图例  
            
        ################ 画X标签 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ############### 画Y标签 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
            
        ################ 将水、牛奶、香蕉的字变大 ################
        axes[countX][countY].tick_params(labelsize=15)
        
        
        ############### 四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黄色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1 

fig.suptitle('RFM-成本获利分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```






![png](img/output_92_1.png)
    


#### 6.2 RFM毛利率分析

- 计算毛利
    - 为了进一步分析清楚, 那些用户身上是赚钱的, 哪些用户是赔钱的,我们可以计算毛利


```python
countfinal['毛利'] = countfinal['clv'] - countfinal['cac']
countfinal['毛利检查'] = np.where(countfinal['毛利']< 0,'#ff5858','#58acff')
```


```python
countfinal
```

<img src="img/rfm_35.png" align='left'/>



- 绘图


```python
#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 #画布Y轴坐标
    for j in recency_label: # 近因
        data = countfinal[(countfinal['recency_cate']==j) & (countfinal['frequency_cate']==i)]
        if data.shape[0] != 0: # 检查这部分有没有数据
            # 下面设定单一小图表
            sns.barplot( #  指定小图标X轴数据
                        y="毛利", #  指定小图标Y轴数据
                        data=data, #绘图使用的DataFrame
                        capsize=.2, 
                        color=data['毛利检查'].values[0],
                        ax=axes[countX, countY]) # 小图表坐标
            
              
        ################ 画X标签 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ###############画Y标签 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
            
        ############### 使所有数据的大小相同 ################
        axes[countX][countY].set_yticks(range(-200,12000,3000))
        
        ############### 四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黄色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1 
fig.suptitle('RFM-毛利分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```




![png](img/output_98_1.png)
    


#### 6.3 RFM 投资回报率分析

- 计算ROI投资回报率
    - 投入一元钱成本会带来多少收益
    - 不同行业ROI不同,我们分析时设定的阈值可以根据自身行业做调整


```python
countfinal['ratio'] = countfinal['clv'] / countfinal['cac']
countfinal['ratio'] = countfinal['ratio'].round(2)
countfinal['ratio'] = countfinal['ratio'].fillna(0)
countfinal['ratio_index'] = np.where(countfinal['ratio']< 3,'#ff5858','#58acff')
countfinal
```

<img src="img/rfm_34.png" align='left'/>

- 绘图


```python
#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 #画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = countfinal[(countfinal['recency_cate']==j) & (countfinal['frequency_cate']==i)]
        if data.shape[0] != 0: # 检查这部分有没有数据
            # 下面设定单一小图表
            sns.barplot( # 指定小图表X轴数据
                        y="ratio", #指定小图标Y轴数据
                        data=data, #数据
                        capsize=.2, 
                        color=data['ratio_index'].values[0],
                        ax=axes[countX, countY]) # 小图表坐标
            
              
         ################ 画X标签 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
         ################ 画y标签 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
        ############### 使所有数据的大小相同 ################
        axes[countX][countY].set_yticks(range(0,10,2))
        
        ############### 四个区块分颜色 ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黄色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色     
        countY += 1 
    countX += 1 
fig.suptitle('RFM-投资回报率分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```




![png](img/output_104_1.png)
    


- 从上图可以看出, 当前业务的ROI主要与购买的频率有关

### 7、RFM营销分析

- 在上一小节中,我们通过ROI分析, 得出某些客群的ROI较低, 接下来我们要尝试把这部分低ROI的用户的营销预算做一些调整,比如减少一半
- 我们把在低ROI客群的营销费用转移到高ROI的客群,我们来试算一下, 经过这样的调整,我们的ROI是否会有提升
- 这里假设不同客群的ROI在试算过程中是固定的


```python
#营销数据计算
total_cac = countfinal['cac'].sum() #总成本
total_clv = countfinal['clv'].sum() #总收益

#将所有转化率低的成本移到高的部分
countfinal['cac'] = np.where(countfinal['ratio'] < 3 , 
                             countfinal['cac']/2, 
                             countfinal['cac'] + ((countfinal[countfinal['ratio'] < 3]['cac']/2).mean())/2 )

countfinal['clv'] = np.where(countfinal['ratio'] < 3 ,
                             countfinal['clv']/2, 
                             countfinal['ratio'] * countfinal['cac'])

countfinal['ratio'] = countfinal['clv'] / countfinal['cac']
```

- 计算新的毛利


```python
countfinal['新毛利'] = countfinal['clv'] - countfinal['cac']
countfinal['新毛利检查'] = np.where(countfinal['新毛利']< 0,'#ffa6a6','#a6ffff')
```


```python
#先设定画布大小
fig, axes = plt.subplots(6, 6,figsize=(25,15))
countX = 0 # 画布X轴坐标
for i in frequency_label[::-1]: # 我们内层的小图是从上向下绘制的, 所以要先绘制频率>5次的组别, 需要将列表倒转遍历
    countY = 0 # 画布Y轴坐标
    for j in recency_label: # 近因
        data = countfinal[(countfinal['recency_cate']==j) & (countfinal['frequency_cate']==i)]
        if data.shape[0] != 0: #  检查这部分有没有数据
            #下面设定单一小图表
            sns.barplot( # 指定小图表X据
                        y="新毛利", # 指定小图标表Y轴数据
                        data=data, #绘图使用的DataFrame
                        capsize=.2, 
                        color=data['新毛利检查'].values[0],
                        ax=axes[countX, countY]) # 小图表坐标
            
            sns.barplot( # 指定小图表X据
                        y="毛利", # 指定小图标表Y轴数据
                        data=data, #绘图使用的DataFrame
                        capsize=.2, 
                        color=data['毛利检查'].values[0],
                        ax=axes[countX, countY]) # 小图表坐标
         ################ 画X标签 ################
        if i == '1 freq':
            axes[countX][countY].set_xlabel(j, fontsize=17)
            
        ############### 画y标签 ################
        if j == '0-7 day':
            axes[countX][countY].set_ylabel( frequency_label[::-1][countX], fontsize=17)
        else:
            axes[countX][countY].set_ylabel('')
            
            
        ################ 将水、牛奶、香蕉的字变大 ################
        axes[countX][countY].tick_params(labelsize=15)
        ############### 使所有数据的大小相同  ################
        axes[countX][countY].set_yticks(range(-200,12000,3000))
        
        ############### 四个区块分颜色  ################
        if countY > 2 and countX > 2:
            axes[countX][countY].set(facecolor="#ffcdd2") #红色
        elif countY > 2 and countX < 3:
            axes[countX][countY].set(facecolor="#FFF9C4") #黄色
        elif countY < 3 and countX > 2:
            axes[countX][countY].set(facecolor="#BBDEFB") #蓝色
        else:
            axes[countX][countY].set(facecolor="#B2DFDB") #绿色
            
        countY += 1 
    countX += 1 
fig.suptitle('RFM-毛利分析图', position=(.5,1), fontsize=35) # 标题
fig.text(0.5, 0.01, '光顾天数', ha='center', va='center', fontsize=20) # X轴标题
fig.text(0.01, 0.5, '购买频率', ha='center', va='center', rotation='vertical', fontsize=20) # Y轴标题
fig.show()
```




![png](img/output_111_1.png)
    


- 上面的图中比较关键的部分是, 我们绘制了两次柱状图, 把柱子较高的先画, 柱子矮的后画, 叠放到一起之后, 就可以看到上面的效果
- 业务解读:
    - 习惯集中购买的客群毛利的提升比较明显, 在营销预算有限的前提下, 我们将预算转移到集中购买的客群中可能会起到更好的效果

### 8、RFM顾客复购分析

#### 8.1 复购分析

- RFM中 recency无法处理的问题，下面两个用户那个用户更活跃？

  ![](img/rfm_40.png)

- 客户活跃度：CAI (Customer Active Index)

  - 评估下面三个用户的活跃程度，如果只是考虑次数， 不考虑间隔， 这三个用户活跃程度一样

  ![](img/rfm_41.png)

  - 但实际上， C用户来的时间间隔越来越短， A的频率比较固定, B的时间间隔越来越长

    ![](img/rfm_42.png)

  - 此时可以考虑加权计算活跃

    - A顾客:![image-20210730031131143](C:\Users\Administrator\Downloads\RFM\img/rfm_44.png)
    - B顾客:![image-20210730031015088](C:\Users\Administrator\Downloads\RFM\img/rfm_43.png)

- 加载数据, 数据条目数更多


```python
orders2 = pd.read_csv('data/orders_2.csv')
orders2.dropna(inplace = True)
```

- RFM处理 代码与之前相同


```python
orders2['values'] = 1
purchase_list2 = orders2.pivot_table(index=['clientId','gender','orderdate'], #分类条件
                          columns='product', # 透视表列名
                          aggfunc=sum, # 计算方式，max, min, mean, sum, len
                          values='values' #值
                          ).fillna(0).reset_index()

##### 频率计算 #####
#计算每个消费者在一定时期内购买产品的次数
purchase_list2['frequency'] = 1
frequency = purchase_list2.groupby("clientId", #分类条件
                                  as_index = False # 分类条件是否要取代Index
                                  )['frequency'].sum() 
# 删除该字段
del purchase_list2['frequency']

# 合并 frequency 
purchase_list2 =purchase_list2.merge(frequency, #即将合并的数据
                                   on = ['clientId'] # 两者时作为
                                   ,how='inner') # 合并的方式

##### 最近一次消费计算 #####
# 假设今天的日期就是数据集中最大的购买日期, 我们以这一天为基准考复购情况
theToday = datetime.datetime.strptime(orders['orderdate'].max(), "%Y-%m-%d")
# 转换日期时间格式
purchase_list2['orderdate'] = pd.to_datetime(purchase_list2['orderdate'])
# 计算theToday距离最后一次消费的时间差
purchase_list2['recency'] =( theToday - purchase_list2['orderdate'] ).astype(str)
# 將'recency'列中的days后缀去掉
purchase_list2['recency'] = purchase_list2['recency'].str.replace('days.*', #想取代的東西
                                                                  '', #取代成的東西
                                                                  regex = True)
# 将'recency'列全部转换成int
purchase_list2['recency'] = purchase_list2['recency'].astype(int)
```




```python
purchase_list2['interval'] = purchase_list2.groupby("clientId", #分类条件
                                  as_index = True # 分类条件是否要取代Index
                                  )['orderdate'].diff()
purchase_list2.dropna(inplace = True)#刪除第一次來本店的数据
purchase_list2['interval'] = purchase_list2['interval'].astype(str) #转换成字符串
purchase_list2['interval'] = purchase_list2['interval'].str.replace('days.*', '').astype(int) #删掉day字段
```

```python
purchase_list2
```

<img src="img/rfm_33.png" align='left'/>

- 计算第几次来店里


```python
purchase_list2['cumsum'] = 1
purchase_list2['cumsum'] = purchase_list2.groupby("clientId")['cumsum'].cumsum()
purchase_list2
```

<img src="img/rfm_32.png" align='left'/>

- 计算活跃指数


```python
#算平均
interval_mean = purchase_list2.groupby("clientId", as_index = False)['interval'].mean()
interval_mean.rename(columns={"interval": "interval_mean"}, inplace = True)
```


```python
interval_mean
```

<img src="img/rfm_31.png" align='left'/>


```python
# 合并平均
purchase_list2 =purchase_list2.merge(interval_mean, # 要合并的资料
                                   on = ['clientId'] # 两张表链接的Key
                                   ,how='inner') # 合并的方式
```


```python
purchase_list2
```

<img src="img/rfm_30.png" align='left'/>


```python
purchase_list2['weighted_average'] = purchase_list2['interval'] * purchase_list2['cumsum'] / (purchase_list2['frequency']*(purchase_list2['frequency'] -1)/2) / purchase_list2['interval_mean']
purchase_list2
```

<img src="img/rfm_29.png" align='left'/>


```python
clientId_weighted_average = purchase_list2.groupby("clientId", as_index = False)['weighted_average'].sum()
clientId_weighted_average['weighted_average'] = 1-clientId_weighted_average['weighted_average']
clientId_weighted_average
```

<img src="img/rfm_28.png" align='left'/>

- 以0作为基准


```python
clientId_weighted_average['back_probability'] = np.where( clientId_weighted_average['weighted_average'] >= 0, 'good','bad')
```


```python
clientId_weighted_average
```

<img src="img/rfm_27.png" align='left'/>

- 换成百分比


```python
clientId_weighted_average['percentage'] = (clientId_weighted_average['weighted_average']  - clientId_weighted_average['weighted_average'].min()) / (clientId_weighted_average['weighted_average'].max() - clientId_weighted_average['weighted_average'].min()) *100
```


```python
clientId_weighted_average
```

<img src="img/rfm_26.png" align='left'/>

#### 8.2 商品推荐清单


```python
recommended_list = purchase_list2.groupby("clientId", #分类条件
                                  as_index = False # 分类条件是否要取代Index
                                  )['banana', 'milk', 'water'].sum() 
```

```python
recommended_list
```

<img src="img/rfm_25.png" align='left'/>


```python
sort=[]
for i in range(len(recommended_list)):
    sort.append(' > '.join(recommended_list[['banana', 'milk', 'water']].iloc[i].sort_values(ascending=False).index.values))
```


```python
clientId_weighted_average['recommended_list']=sort
```


```python
#回来算Recency、Frequency
clientId_weighted_average['recency'] = purchase_list.groupby("clientId", #分类条件
                                                              as_index = False # 分类条件是否要取代Index
                                                              )['recency'].min()[['recency']] 
clientId_weighted_average['frequency'] = purchase_list.groupby("clientId", #分类条件
                                                              as_index = False # 分类条件是否要取代Index
                                                              )['frequency'].min()['frequency'] # 
clientId_weighted_average
```

<img src="img/rfm_24.png" align='left'/>




# 2 Pandas 数据结构

## 学习目标

- 掌握Series的常用属性及方法
- 掌握DataFrame的常用属性及方法
- 掌握更改Series和DataFrame的方法
- 掌握如何导入导出数据

## 1 Series和DataFrame

- DataFrame和Series是Pandas最基本的两种数据结构
- DataFrame用来处理结构化数据（SQL数据表，Excel表格）
- Series用来处理单列数据，也可以把DataFrame看作由Series对象组成的字典或集合

### 1.1 创建Series

-  在Pandas中，Series是一维容器，Series表示DataFrame的每一列

  - 可以把DataFrame看作由Series对象组成的字典，其中key是列名，值是Series
  - Series和Python中的列表非常相似，但是它的每个元素的数据类型必须相同

- 创建 Series 的最简单方法是传入一个Python列表，如果传入的数据类型不统一，最终的dtype通常是object

  ```python
  import pandas as pd
  s = pd.Series(['banana',42])
  print(s)
  ```

  >  <font color = red>输出结果</font>
  >
  > ```python
  > 0    banana
  > 1        42
  > dtype: object
  > ```

  - 上面的结果中，左边显示的0,1是Series的索引

- 创建Series时，可以通过index参数 来指定行索引

  ```python
  s = pd.Series(['Wes McKinney','Male'],index = ['Name','Gender'])
  print(s)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >Name      Wes McKinney
  >Gender            Male
  >dtype: object
  >```

### 1.2 创建 DataFrame

- 可以使用字典来创建DataFrame

  ```python
  name_list = pd.DataFrame(
      {'Name':['Tom','Bob'],
       'Occupation':['Teacher','IT Engineer'],
       'age':[28,36]})
  print(name_list)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >   Name   Occupation  age
  >0  Tom      Teacher   28
  >1   Bob  IT Engineer   36
  >```

- 创建DataFrame的时候可以使用colums参数指定列的顺序，也可以使用index来指定行索引

  ```python
  name_list = pd.DataFrame(data = {'Occupation':['Teacher','IT Engineer'],'Age':[28,36]},columns=['Age','Occupation'],index=['Tom','Bob'])
  print(name_list)
  ```

  ><font color = red>输出结果</font>
  >
  >```python
  >     Age   Occupation
  >Tom   28      Teacher
  >Bob    36  IT Engineer
  >```

## 2 Series 常用操作

### 2.1 Series常用属性

- 使用 DataFrame的loc 属性获取数据集里的一行，就会得到一个Series对象

  ```python
  data = pd.read_csv('data/nobel_prizes.csv',index_col='id')
  data.head()
  ```

  ><font color = red>输出结果</font>
  
  |   id | year | category  | overallMotivation | firstname | surname  | motivation                                        | share |
  | ---: | :--- | :-------- | :---------------- | :-------- | :------- | :------------------------------------------------ | :---- |
  |  941 | 2017 | physics   | NaN               | Rainer    | Weiss    | "for decisive contributions to the LIGO detect... | 2     |
  |  942 | 2017 | physics   | NaN               | Barry C.  | Barish   | "for decisive contributions to the LIGO detect... | 4     |
  |  943 | 2017 | physics   | NaN               | Kip S.    | Thorne   | "for decisive contributions to the LIGO detect... | 4     |
  |  944 | 2017 | chemistry | NaN               | Jacques   | Dubochet | "for developing cryo-electron microscopy for t... | 3     |
  |  945 | 2017 | chemistry | NaN               | Joachim   | Frank    | "for developing cryo-electron microscopy for t... | 3     |
  
- 使用行索引标签选择一条记录

  ```python
  first_row = data.loc[941]
  type(first_row)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >pandas.core.series.Series
  >```

  ```python
  print(first_row)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >year                                                              2017
  >category                                                       physics
  >overallMotivation                                                  NaN
  >firstname                                                       Rainer
  >surname                                                          Weiss
  >motivation           "for decisive contributions to the LIGO detect...
  >share                                                                2
  >Name: 941, dtype: object
  >```

- 可以通过 index 和 values属性获取行索引和值

  ```python
  first_row.index
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >Index(['year', 'category', 'overallMotivation', 'firstname', 'surname',
  >       'motivation', 'share'],
  >      dtype='object')
  >```

  ```python
  print(first_row.values)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >[2017 'physics' nan 'Rainer' 'Weiss'
  > '"for decisive contributions to the LIGO detector and the observation of gravitational waves"'
  > 2]
  >```

- Series的keys方法，作用个index属性一样

  ```python
  data.keys()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >Index(['year', 'category', 'overallMotivation', 'firstname', 'surname',
  >       'motivation', 'share'],
  >      dtype='object')
  >```

- Series的一些属性

  <table>
    <tr>
      <td>属性</td><td>说明</td>
    </tr>
    <tr>
      <td>loc</td><td>使用索引值取子集</td>
    </tr>
    <tr>
      <td>iloc</td><td>使用索引位置取子集</td>
    </tr>
    <tr>
      <td>dtype或dtypes</td><td>Series内容的类型</td>
    </tr>
     <tr>
      <td>shape</td><td>数据的维数</td>
    </tr>
    <tr>
      <td>size</td><td>Series中元素的数量</td>
    </tr>
    <tr>
      <td>values</td><td>Series的值</td>
    </tr>
  </table>

### 2.2 Series常用方法

- 针对数值型的Series，可以进行常见计算

  ```python
  share = data.share  # 从DataFrame中 获取Share列（几人获奖）返回Series
  share.mean()      #计算几人获奖的平均值
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >1.982665222101842
  >```

  ```python
  share.max() # 计算最大值
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >4
  >```

  ```python
  share.min() # 计算最小值
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >1
  >```

  ```python
  share.std() # 计算标准差
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0.9324952202244597
  >```

- 通过value_counts()方法，可以返回不同值的条目数量

  ```python
  movie = pd.read_csv('data/movie.csv')    # 加载电影数据
  director = movie['director_name']   # 从电影数据中获取导演名字 返回Series
  actor_1_fb_likes = movie['actor_1_facebook_likes'] # 从电影数据中取出主演的facebook点赞数
  director.head()  #查看导演Series数据
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0        James Cameron
  >1       Gore Verbinski
  >2           Sam Mendes
  >3    Christopher Nolan
  >4          Doug Walker
  >Name: director_name, dtype: object
  >```

  ```python
  actor_1_fb_likes.head() #查看主演的facebook点赞数
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0     1000.0
  >1    40000.0
  >2    11000.0
  >3    27000.0
  >4      131.0
  >Name: actor_1_facebook_likes, dtype: float64
  >```

  ```python
  pd.set_option('max_rows', 8) # 设置最多显示8行
  director.value_counts()      # 统计不同导演指导的电影数量
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >Steven Spielberg    26
  >Woody Allen         22
  >Clint Eastwood      20
  >Martin Scorsese     20
  >                    ..
  >Gavin Wiesen         1
  >Andrew Morahan       1
  >Luca Guadagnino      1
  >Richard Montoya      1
  >Name: director_name, Length: 2397, dtype: int64
  >```

  ```python
  actor_1_fb_likes.value_counts()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >1000.0     436
  >11000.0    206
  >2000.0     189
  >3000.0     150
  >          ... 
  >216.0        1
  >859.0        1
  >225.0        1
  >334.0        1
  >Name: actor_1_facebook_likes, Length: 877, dtype: int64
  >```

- 通过count()方法可以返回有多少非空值

  ```python
  director.count() 
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >4814
  >```

  ```python
  director.shape
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >(4916,)
  >```

- 通过describe()方法打印描述信息

  ```python
  actor_1_fb_likes.describe()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >count      4909.000000
  >mean       6494.488491
  >std       15106.986884
  >min           0.000000
  >25%         607.000000
  >50%         982.000000
  >75%       11000.000000
  >max      640000.000000
  >Name: actor_1_facebook_likes, dtype: float64
  >```

  ```python
  director.describe()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >count                 4814
  >unique                2397
  >top       Steven Spielberg
  >freq                    26
  >Name: director_name, dtype: object
  >```

- Series的一些方法

  <table>
    <tr>
      <td>方法</td><td>说明</td>
    </tr>
    <tr>
      <td>append</td><td>连接两个或多个Series</td>
    </tr>
    <tr>
      <td>corr</td><td>计算与另一个Series的相关系数</td>
    </tr>
    <tr>
      <td>cov</td><td>计算与另一个Series的协方差</td>
    </tr>
    <tr>
      <td>describe</td><td>计算常见统计量</td>
    </tr>
     <tr>
      <td>drop_duplicates</td><td>返回去重之后的Series</td>
    </tr>
    <tr>
      <td>equals</td><td>判断两个Series是否相同</td>
    </tr>
    <tr>
      <td>hist</td><td>绘制直方图</td>
    </tr>
    <tr>
      <td>isin</td><td>Series中是否包含某些值</td>
    </tr>
    <tr>
      <td>min</td><td>返回最小值</td>
    </tr>
    <tr>
      <td>max</td><td>返回最大值</td>
    </tr>
    <tr>
      <td>mean</td><td>返回算术平均值</td>
    </tr>
    <tr>
      <td>median</td><td>返回中位数</td>
    </tr>
    <tr>
      <td>mode</td><td>返回众数</td>
    </tr>
    <tr>
      <td>quantile</td><td>返回指定位置的分位数</td>
    </tr>
    <tr>
      <td>replace</td><td>用指定值代替Series中的值</td>
    </tr>
    <tr>
      <td>sample</td><td>返回Series的随机采样值</td>
    </tr>
    <tr>
      <td>sort_values</td><td>对值进行排序</td>
    </tr>
    <tr>
      <td>to_frame</td><td>把Series转换为DataFrame</td>
    </tr>
    <tr>
      <td>unique</td><td>去重返回数组</td>
    </tr>
  </table>

### 2.3 Series的布尔索引

- 从Series中获取满足某些条件的数据，可以使用布尔索引

  ```python
  scientists = pd.read_csv('data/scientists.csv')
  print(scientists)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >                   Name        Born        Died  Age          Occupation
  >0     Rosaline Franklin  1920-07-25  1958-04-16   37             Chemist
  >1        William Gosset  1876-06-13  1937-10-16   61        Statistician
  >2  Florence Nightingale  1820-05-12  1910-08-13   90               Nurse
  >3           Marie Curie  1867-11-07  1934-07-04   66             Chemist
  >4         Rachel Carson  1907-05-27  1964-04-14   56           Biologist
  >5             John Snow  1813-03-15  1858-06-16   45           Physician
  >6           Alan Turing  1912-06-23  1954-06-07   41  Computer Scientist
  >7          Johann Gauss  1777-04-30  1855-02-23   77       Mathematician
  >```

  - 获取大于平均年龄的结果

  ```python
  ages = scientists['Age']
  ages.mean()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >59.125
  >```

  ```python
  print(ages[ages>ages.mean()])
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >1    61
  >2    90
  >3    66
  >7    77
  >Name: Age, dtype: int64
  >```

  - ages>ages.mean() 分析返回结果

  ```python
  print(ages>ages.mean())
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0    False
  >1     True
  >2     True
  >3     True
  >4    False
  >5    False
  >6    False
  >7     True
  >Name: Age, dtype: bool
  >```

  - 从上面结果可以得知，从Series中获取部分数据，可以通过标签，索引，也可以传入布尔值的列表

  ```python
  #手动创建布尔值列表
  bool_values = [False,True,True,False,False,False,False,False]
  ages[bool_values]
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >1    61
  >2    90
  >Name: Age, dtype: int64
  >```

### 2.4 Series 的运算

- Series和数值型变量计算时，变量会与Series中的每个元素逐一进行计算

  ```python
  ages+100
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0    137
  >1    161
  >2    190
  >3    166
  >4    156
  >5    145
  >6    141
  >7    177
  >Name: Age, dtype: int64
  >```

  ```python
  ages*2
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0     74
  >1    122
  >2    180
  >3    132
  >4    112
  >5     90
  >6     82
  >7    154
  >Name: Age, dtype: int64
  >```

- 两个Series之间进行计算，会根据索引进行。索引不同的元素最终计算的结果会填充成缺失值，用NaN表示

  ```python
  ages + pd.Series([1,100])
  ```

  ><font color = red>输出结果</font>
  >
  >```python
  >0     38.0
  >1    161.0
  >2      NaN
  >3      NaN
  >4      NaN
  >5      NaN
  >6      NaN
  >7      NaN
  >dtype: float64
  >```

  ```python
  ages * pd.Series([1,100])
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >0      37.0
  >1    6100.0
  >2       NaN
  >3       NaN
  >4       NaN
  >5       NaN
  >6       NaN
  >7       NaN
  >dtype: float64
  >```


## 3 DataFrame常用操作

### 3.1 DataFrame的常用属性和方法

- DataFrame是Pandas中最常见的对象，Series数据结构的许多属性和方法在DataFrame中也一样适用

  ```python
  movie = pd.read_csv('data/movie.csv')
   # 打印行数和列数
  movie.shape 
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >(4916, 28)
  >```

  ```python
  # 打印数据的个数
  movie.size
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >137648
  >```

  ```python
  # 该数据集的维度
  movie.ndim
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >2
  >```

  ```python
  # 该数据集的长度
  len(movie)
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >4916
  >```

  ```python
  # 各个列的值的个数
  movie.count()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >color                     4897
  >director_name             4814
  >num_critic_for_reviews    4867
  >duration                  4901
  >                          ... 
  >actor_2_facebook_likes    4903
  >imdb_score                4916
  >aspect_ratio              4590
  >movie_facebook_likes      4916
  >Length: 28, dtype: int64
  >```

  ```python
  # 各列的最小值
  movie.min()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >num_critic_for_reviews     1.00
  >duration                   7.00
  >director_facebook_likes    0.00
  >actor_3_facebook_likes     0.00
  >                           ... 
  >actor_2_facebook_likes     0.00
  >imdb_score                 1.60
  >aspect_ratio               1.18
  >movie_facebook_likes       0.00
  >Length: 16, dtype: float64
  >```

  ```python
  movie.describe()
  ```

  ><font color = red>输出结果</font>
  >
  >|       | num_critic_for_reviews |    duration | director_facebook_likes | actor_3_facebook_likes | actor_1_facebook_likes |        gross | num_voted_users | cast_total_facebook_likes | facenumber_in_poster | num_user_for_reviews |       budget |  title_year | actor_2_facebook_likes |  imdb_score | aspect_ratio | movie_facebook_likes |
  >| ----: | ---------------------: | ----------: | ----------------------: | ---------------------: | ---------------------: | -----------: | --------------: | ------------------------: | -------------------: | -------------------: | -----------: | ----------: | ---------------------: | ----------: | -----------: | -------------------: |
  >| count |            4867.000000 | 4901.000000 |             4814.000000 |            4893.000000 |            4909.000000 | 4.054000e+03 |    4.916000e+03 |               4916.000000 |          4903.000000 |          4895.000000 | 4.432000e+03 | 4810.000000 |            4903.000000 | 4916.000000 |  4590.000000 |          4916.000000 |
  >|  mean |             137.988905 |  107.090798 |              691.014541 |             631.276313 |            6494.488491 | 4.764451e+07 |    8.264492e+04 |               9579.815907 |             1.377320 |           267.668846 | 3.654749e+07 | 2002.447609 |            1621.923516 |    6.437429 |     2.222349 |          7348.294142 |
  >|   std |             120.239379 |   25.286015 |             2832.954125 |            1625.874802 |           15106.986884 | 6.737255e+07 |    1.383222e+05 |              18164.316990 |             2.023826 |           372.934839 | 1.002427e+08 |   12.453977 |            4011.299523 |    1.127802 |     1.402940 |         19206.016458 |
  >|   min |               1.000000 |    7.000000 |                0.000000 |               0.000000 |               0.000000 | 1.620000e+02 |    5.000000e+00 |                  0.000000 |             0.000000 |             1.000000 | 2.180000e+02 | 1916.000000 |               0.000000 |    1.600000 |     1.180000 |             0.000000 |
  >|   25% |              49.000000 |   93.000000 |                7.000000 |             132.000000 |             607.000000 | 5.019656e+06 |    8.361750e+03 |               1394.750000 |             0.000000 |            64.000000 | 6.000000e+06 | 1999.000000 |             277.000000 |    5.800000 |     1.850000 |             0.000000 |
  >|   50% |             108.000000 |  103.000000 |               48.000000 |             366.000000 |             982.000000 | 2.504396e+07 |    3.313250e+04 |               3049.000000 |             1.000000 |           153.000000 | 1.985000e+07 | 2005.000000 |             593.000000 |    6.600000 |     2.350000 |           159.000000 |
  >|   75% |             191.000000 |  118.000000 |              189.750000 |             633.000000 |           11000.000000 | 6.110841e+07 |    9.377275e+04 |              13616.750000 |             2.000000 |           320.500000 | 4.300000e+07 | 2011.000000 |             912.000000 |    7.200000 |     2.350000 |          2000.000000 |
  >|   max |             813.000000 |  511.000000 |            23000.000000 |           23000.000000 |          640000.000000 | 7.605058e+08 |    1.689764e+06 |             656730.000000 |            43.000000 |          5060.000000 | 4.200000e+09 | 2016.000000 |          137000.000000 |    9.500000 |    16.000000 |        349000.000000 |

### 3.2 DataFrame的布尔索引

- 同Series一样，DataFrame也可以使用布尔索引获取数据子集。

  ```python
  # 使用布尔索引获取部分数据行
  movie[movie['duration']>movie['duration'].mean()]
  ```

  ><font color = red>输出结果</font>
  >
  >|      | color |     director_name | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes |     actor_2_name | actor_1_facebook_likes |       gross |                             genres |  ... | num_user_for_reviews | language | country | content_rating |      budget | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| ---: | ----: | ----------------: | ---------------------: | -------: | ----------------------: | ---------------------: | ---------------: | ---------------------: | ----------: | ---------------------------------: | ---: | -------------------: | -------: | ------: | -------------: | ----------: | ---------: | ---------------------: | ---------: | -----------: | -------------------: |
  >|    0 | Color |     James Cameron |                  723.0 |    178.0 |                     0.0 |                  855.0 | Joel David Moore |                 1000.0 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi |  ... |               3054.0 |  English |     USA |          PG-13 | 237000000.0 |     2009.0 |                  936.0 |        7.9 |         1.78 |                33000 |
  >|    1 | Color |    Gore Verbinski |                  302.0 |    169.0 |                   563.0 |                 1000.0 |    Orlando Bloom |                40000.0 | 309404152.0 |         Action\|Adventure\|Fantasy |  ... |               1238.0 |  English |     USA |          PG-13 | 300000000.0 |     2007.0 |                 5000.0 |        7.1 |         2.35 |                    0 |
  >|    2 | Color |        Sam Mendes |                  602.0 |    148.0 |                     0.0 |                  161.0 |     Rory Kinnear |                11000.0 | 200074175.0 |        Action\|Adventure\|Thriller |  ... |                994.0 |  English |      UK |          PG-13 | 245000000.0 |     2015.0 |                  393.0 |        6.8 |         2.35 |                85000 |
  >|    3 | Color | Christopher Nolan |                  813.0 |    164.0 |                 22000.0 |                23000.0 |   Christian Bale |                27000.0 | 448130642.0 |                   Action\|Thriller |  ... |               2701.0 |  English |     USA |          PG-13 | 250000000.0 |     2012.0 |                23000.0 |        8.5 |         2.35 |               164000 |
  >|  ... |   ... |               ... |                    ... |      ... |                     ... |                    ... |              ... |                    ... |         ... |                                ... |  ... |                  ... |      ... |     ... |            ... |         ... |        ... |                    ... |        ... |          ... |                  ... |
  >| 4893 |   NaN |   Brandon Landers |                    NaN |    143.0 |                     8.0 |                    8.0 |  Alana Kaniewski |                  720.0 |         NaN |            Drama\|Horror\|Thriller |  ... |                  8.0 |  English |     USA |            NaN |     17350.0 |     2011.0 |                   19.0 |        3.0 |          NaN |                   33 |
  >| 4898 | Color |       John Waters |                   73.0 |    108.0 |                     0.0 |                  105.0 |       Mink Stole |                  462.0 |    180483.0 |              Comedy\|Crime\|Horror |  ... |                183.0 |  English |     USA |          NC-17 |     10000.0 |     1972.0 |                  143.0 |        6.1 |         1.37 |                    0 |
  >| 4899 | Color |   Olivier Assayas |                   81.0 |    110.0 |                   107.0 |                   45.0 |   Béatrice Dalle |                  576.0 |    136007.0 |              Drama\|Music\|Romance |  ... |                 39.0 |   French |  France |              R |      4500.0 |     2004.0 |                  133.0 |        6.9 |         2.35 |                  171 |
  >| 4902 | Color |  Kiyoshi Kurosawa |                   78.0 |    111.0 |                    62.0 |                    6.0 |    Anna Nakagawa |                   89.0 |     94596.0 |   Crime\|Horror\|Mystery\|Thriller |  ... |                 50.0 | Japanese |   Japan |            NaN |   1000000.0 |     1997.0 |                   13.0 |        7.4 |         1.85 |                  817 |
  >
  >2006 rows × 28 columns

- 可以传入布尔值的列表，来获取部分数据，True所对应的数据会被保留

  ```python
  movie.head()[[True,True,False,True,False]]
  ```

  ><font color = red>输出结果</font>
  >
  >| color | director_name | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes | actor_2_name | actor_1_facebook_likes |   gross |      genres |                                ... | num_user_for_reviews | language | country | content_rating | budget |  title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |        |
  >| ----: | ------------: | ---------------------: | -------: | ----------------------: | ---------------------: | -----------: | ---------------------: | ------: | ----------: | ---------------------------------: | -------------------: | -------: | ------: | -------------: | -----: | ----------: | ---------------------: | ---------: | -----------: | -------------------: | ------ |
  >|     0 |         Color |          James Cameron |    723.0 |                   178.0 |                    0.0 |        855.0 |       Joel David Moore |  1000.0 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi |                  ... |   3054.0 | English |            USA |  PG-13 | 237000000.0 |                 2009.0 |      936.0 |          7.9 |                 1.78 | 33000  |
  >|     1 |         Color |         Gore Verbinski |    302.0 |                   169.0 |                  563.0 |       1000.0 |          Orlando Bloom | 40000.0 | 309404152.0 |         Action\|Adventure\|Fantasy |                  ... |   1238.0 | English |            USA |  PG-13 | 300000000.0 |                 2007.0 |     5000.0 |          7.1 |                 2.35 | 0      |
  >|     3 |         Color |      Christopher Nolan |    813.0 |                   164.0 |                22000.0 |      23000.0 |         Christian Bale | 27000.0 | 448130642.0 |                   Action\|Thriller |                  ... |   2701.0 | English |            USA |  PG-13 | 250000000.0 |                 2012.0 |    23000.0 |          8.5 |                 2.35 | 164000 |
  >
  >3 rows × 28 columns

### 3.3 DataFrame的运算

- 当DataFrame和数值进行运算时，DataFrame中的每一个元素会分别和数值进行运算

  ```python
  scientists*2
  ```

  ><font color = red>输出结果</font>
  >
  >|      |                                     Name |                 Born |                 Died |  Age |                           Occupation |
  >| ---: | ---------------------------------------: | -------------------: | -------------------: | ---: | -----------------------------------: |
  >|    0 |       Rosaline FranklinRosaline Franklin | 1920-07-251920-07-25 | 1958-04-161958-04-16 |   74 |                       ChemistChemist |
  >|    1 |             William GossetWilliam Gosset | 1876-06-131876-06-13 | 1937-10-161937-10-16 |  122 |             StatisticianStatistician |
  >|    2 | Florence NightingaleFlorence Nightingale | 1820-05-121820-05-12 | 1910-08-131910-08-13 |  180 |                           NurseNurse |
  >|    3 |                   Marie CurieMarie Curie | 1867-11-071867-11-07 | 1934-07-041934-07-04 |  132 |                       ChemistChemist |
  >|    4 |               Rachel CarsonRachel Carson | 1907-05-271907-05-27 | 1964-04-141964-04-14 |  112 |                   BiologistBiologist |
  >|    5 |                       John SnowJohn Snow | 1813-03-151813-03-15 | 1858-06-161858-06-16 |   90 |                   PhysicianPhysician |
  >|    6 |                   Alan TuringAlan Turing | 1912-06-231912-06-23 | 1954-06-071954-06-07 |   82 | Computer ScientistComputer Scientist |
  >|    7 |                 Johann GaussJohann Gauss | 1777-04-301777-04-30 | 1855-02-231855-02-23 |  154 |           MathematicianMathematician |

  - 在上面的例子中，数据都是字符串类型的，所有字符串都被复制了一份
  - 上面的例子，如果做其它计算会报错

- 两个DataFrame之间进行计算，会根据索引进行对应计算

  ```python
  scientists+scientists
  ```

  ><font color = red>输出结果</font>
  >
  >| Name |                                     Born |                 Died |                  Age | Occupation |                                      |
  >| ---: | ---------------------------------------: | -------------------: | -------------------: | ---------: | ------------------------------------ |
  >|    0 |       Rosaline FranklinRosaline Franklin | 1920-07-251920-07-25 | 1958-04-161958-04-16 |         74 | ChemistChemist                       |
  >|    1 |             William GossetWilliam Gosset | 1876-06-131876-06-13 | 1937-10-161937-10-16 |        122 | StatisticianStatistician             |
  >|    2 | Florence NightingaleFlorence Nightingale | 1820-05-121820-05-12 | 1910-08-131910-08-13 |        180 | NurseNurse                           |
  >|    3 |                   Marie CurieMarie Curie | 1867-11-071867-11-07 | 1934-07-041934-07-04 |        132 | ChemistChemist                       |
  >|    4 |               Rachel CarsonRachel Carson | 1907-05-271907-05-27 | 1964-04-141964-04-14 |        112 | BiologistBiologist                   |
  >|    5 |                       John SnowJohn Snow | 1813-03-151813-03-15 | 1858-06-161858-06-16 |         90 | PhysicianPhysician                   |
  >|    6 |                   Alan TuringAlan Turing | 1912-06-231912-06-23 | 1954-06-071954-06-07 |         82 | Computer ScientistComputer Scientist |
  >|    7 |                 Johann GaussJohann Gauss | 1777-04-301777-04-30 | 1855-02-231855-02-23 |        154 | MathematicianMathematician           |

  - 两个DataFrame数据条目数不同时，会根据索引进行计算，索引不匹配的会返回NaN

  ```python
  first_half = scientists[:4]
  scientists+first_half
  ```

  ><font color = red>输出结果</font>
  >
  >| Name |                                     Born |                 Died |                  Age | Occupation |                          |
  >| ---: | ---------------------------------------: | -------------------: | -------------------: | ---------: | ------------------------ |
  >|    0 |       Rosaline FranklinRosaline Franklin | 1920-07-251920-07-25 | 1958-04-161958-04-16 |       74.0 | ChemistChemist           |
  >|    1 |             William GossetWilliam Gosset | 1876-06-131876-06-13 | 1937-10-161937-10-16 |      122.0 | StatisticianStatistician |
  >|    2 | Florence NightingaleFlorence Nightingale | 1820-05-121820-05-12 | 1910-08-131910-08-13 |      180.0 | NurseNurse               |
  >|    3 |                   Marie CurieMarie Curie | 1867-11-071867-11-07 | 1934-07-041934-07-04 |      132.0 | ChemistChemist           |
  >|    4 |                                      NaN |                  NaN |                  NaN |        NaN | NaN                      |
  >|    5 |                                      NaN |                  NaN |                  NaN |        NaN | NaN                      |
  >|    6 |                                      NaN |                  NaN |                  NaN |        NaN | NaN                      |
  >|    7 |                                      NaN |                  NaN |                  NaN |        NaN | NaN                      |
  

## 4 修改Series和DataFrame

### 4.1 给行索引命名

- 加载数据文件时，如果不指定行索引，Pandas会自动加上从0开始的索引，可以通过set_index()方法重新设置行索引的名字
  ```python
  movie = pd.read_csv('data/movie.csv')
  movie
  ```

  ><font color = red>输出结果</font>
  >
  >|      | color |     director_name | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes |     actor_2_name | actor_1_facebook_likes |       gross |                             genres |  ... | num_user_for_reviews | language | country | content_rating |      budget | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| ---: | ----: | ----------------: | ---------------------: | -------: | ----------------------: | ---------------------: | ---------------: | ---------------------: | ----------: | ---------------------------------: | ---: | -------------------: | -------: | ------: | -------------: | ----------: | ---------: | ---------------------: | ---------: | -----------: | -------------------- |
  >|    0 | Color |     James Cameron |                  723.0 |    178.0 |                     0.0 |                  855.0 | Joel David Moore |                 1000.0 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi |  ... |               3054.0 |  English |     USA |          PG-13 | 237000000.0 |     2009.0 |                  936.0 |        7.9 |         1.78 | 33000                |
  >|    1 | Color |    Gore Verbinski |                  302.0 |    169.0 |                   563.0 |                 1000.0 |    Orlando Bloom |                40000.0 | 309404152.0 |         Action\|Adventure\|Fantasy |  ... |               1238.0 |  English |     USA |          PG-13 | 300000000.0 |     2007.0 |                 5000.0 |        7.1 |         2.35 | 0                    |
  >|    2 | Color |        Sam Mendes |                  602.0 |    148.0 |                     0.0 |                  161.0 |     Rory Kinnear |                11000.0 | 200074175.0 |        Action\|Adventure\|Thriller |  ... |                994.0 |  English |      UK |          PG-13 | 245000000.0 |     2015.0 |                  393.0 |        6.8 |         2.35 | 85000                |
  >|    3 | Color | Christopher Nolan |                  813.0 |    164.0 |                 22000.0 |                23000.0 |   Christian Bale |                27000.0 | 448130642.0 |                   Action\|Thriller |  ... |               2701.0 |  English |     USA |          PG-13 | 250000000.0 |     2012.0 |                23000.0 |        8.5 |         2.35 | 164000               |
  >|  ... |   ... |               ... |                    ... |      ... |                     ... |                    ... |              ... |                    ... |         ... |                                ... |  ... |                  ... |      ... |     ... |            ... |         ... |        ... |                    ... |        ... |          ... | ...                  |
  >| 4912 | Color |               NaN |                   43.0 |     43.0 |                     NaN |                  319.0 |    Valorie Curry |                  841.0 |         NaN |    Crime\|Drama\|Mystery\|Thriller |  ... |                359.0 |  English |     USA |          TV-14 |         NaN |        NaN |                  593.0 |        7.5 |        16.00 | 32000                |
  >| 4913 | Color |  Benjamin Roberds |                   13.0 |     76.0 |                     0.0 |                    0.0 |    Maxwell Moody |                    0.0 |         NaN |            Drama\|Horror\|Thriller |  ... |                  3.0 |  English |     USA |            NaN |      1400.0 |     2013.0 |                    0.0 |        6.3 |          NaN | 16                   |
  >| 4914 | Color |       Daniel Hsia |                   14.0 |    100.0 |                     0.0 |                  489.0 |    Daniel Henney |                  946.0 |     10443.0 |             Comedy\|Drama\|Romance |  ... |                  9.0 |  English |     USA |          PG-13 |         NaN |     2012.0 |                  719.0 |        6.3 |         2.35 | 660                  |
  >| 4915 | Color |          Jon Gunn |                   43.0 |     90.0 |                    16.0 |                   16.0 | Brian Herzlinger |                   86.0 |     85222.0 |                        Documentary |  ... |                 84.0 |  English |     USA |             PG |      1100.0 |     2004.0 |                   23.0 |        6.6 |         1.85 | 456                  |
  >
  >4916 rows × 28 columns
  ```python
  movie2 = movie.set_index('movie_title')
  movie2
  ```

  ><font color = red>输出结果</font>
  >
  >|                              movie_title | color | director_name     | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes | actor_2_name     | actor_1_facebook_likes | gross       | genres                             | ...  | num_user_for_reviews | language | country | content_rating | budget      | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| ---------------------------------------: | :---- | :---------------- | :--------------------- | :------- | :---------------------- | :--------------------- | :--------------- | :--------------------- | :---------- | :--------------------------------- | :--- | :------------------- | :------- | :------ | :------------- | :---------- | :--------- | :--------------------- | :--------- | :----------- | :------------------- |
  >|                                   Avatar | Color | James Cameron     | 723.0                  | 178.0    | 0.0                     | 855.0                  | Joel David Moore | 1000.0                 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi | ...  | 3054.0               | English  | USA     | PG-13          | 237000000.0 | 2009.0     | 936.0                  | 7.9        | 1.78         | 33000                |
  >| Pirates of the Caribbean: At World's End | Color | Gore Verbinski    | 302.0                  | 169.0    | 563.0                   | 1000.0                 | Orlando Bloom    | 40000.0                | 309404152.0 | Action\|Adventure\|Fantasy         | ...  | 1238.0               | English  | USA     | PG-13          | 300000000.0 | 2007.0     | 5000.0                 | 7.1        | 2.35         | 0                    |
  >|                                  Spectre | Color | Sam Mendes        | 602.0                  | 148.0    | 0.0                     | 161.0                  | Rory Kinnear     | 11000.0                | 200074175.0 | Action\|Adventure\|Thriller        | ...  | 994.0                | English  | UK      | PG-13          | 245000000.0 | 2015.0     | 393.0                  | 6.8        | 2.35         | 85000                |
  >|                    The Dark Knight Rises | Color | Christopher Nolan | 813.0                  | 164.0    | 22000.0                 | 23000.0                | Christian Bale   | 27000.0                | 448130642.0 | Action\|Thriller                   | ...  | 2701.0               | English  | USA     | PG-13          | 250000000.0 | 2012.0     | 23000.0                | 8.5        | 2.35         | 164000               |
  >|                                      ... | ...   | ...               | ...                    | ...      | ...                     | ...                    | ...              | ...                    | ...         | ...                                | ...  | ...                  | ...      | ...     | ...            | ...         | ...        | ...                    | ...        | ...          | ...                  |
  >|                            The Following | Color | NaN               | 43.0                   | 43.0     | NaN                     | 319.0                  | Valorie Curry    | 841.0                  | NaN         | Crime\|Drama\|Mystery\|Thriller    | ...  | 359.0                | English  | USA     | TV-14          | NaN         | NaN        | 593.0                  | 7.5        | 16.00        | 32000                |
  >|                     A Plague So Pleasant | Color | Benjamin Roberds  | 13.0                   | 76.0     | 0.0                     | 0.0                    | Maxwell Moody    | 0.0                    | NaN         | Drama\|Horror\|Thriller            | ...  | 3.0                  | English  | USA     | NaN            | 1400.0      | 2013.0     | 0.0                    | 6.3        | NaN          | 16                   |
  >|                         Shanghai Calling | Color | Daniel Hsia       | 14.0                   | 100.0    | 0.0                     | 489.0                  | Daniel Henney    | 946.0                  | 10443.0     | Comedy\|Drama\|Romance             | ...  | 9.0                  | English  | USA     | PG-13          | NaN         | 2012.0     | 719.0                  | 6.3        | 2.35         | 660                  |
  >|                        My Date with Drew | Color | Jon Gunn          | 43.0                   | 90.0     | 16.0                    | 16.0                   | Brian Herzlinger | 86.0                   | 85222.0     | Documentary                        | ...  | 84.0                 | English  | USA     | PG             | 1100.0      | 2004.0     | 23.0                   | 6.6        | 1.85         | 456                  |
  >
  >4916 rows × 27 columns

- 加载数据的时候，可以通过通过index_col参数，指定使用某一列数据作为行索引
  ```python
  pd.read_csv('data/movie.csv', index_col='movie_title')
  ```

  ><font color = red>输出结果</font>
  >
  >|                              movie_title | color | director_name     | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes | actor_2_name     | actor_1_facebook_likes | gross       | genres                             | ...  | num_user_for_reviews | language | country | content_rating | budget      | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| ---------------------------------------: | :---- | :---------------- | :--------------------- | :------- | :---------------------- | :--------------------- | :--------------- | :--------------------- | :---------- | :--------------------------------- | :--- | :------------------- | :------- | :------ | :------------- | :---------- | :--------- | :--------------------- | :--------- | :----------- | :------------------- |
  >|                                   Avatar | Color | James Cameron     | 723.0                  | 178.0    | 0.0                     | 855.0                  | Joel David Moore | 1000.0                 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi | ...  | 3054.0               | English  | USA     | PG-13          | 237000000.0 | 2009.0     | 936.0                  | 7.9        | 1.78         | 33000                |
  >| Pirates of the Caribbean: At World's End | Color | Gore Verbinski    | 302.0                  | 169.0    | 563.0                   | 1000.0                 | Orlando Bloom    | 40000.0                | 309404152.0 | Action\|Adventure\|Fantasy         | ...  | 1238.0               | English  | USA     | PG-13          | 300000000.0 | 2007.0     | 5000.0                 | 7.1        | 2.35         | 0                    |
  >|                                  Spectre | Color | Sam Mendes        | 602.0                  | 148.0    | 0.0                     | 161.0                  | Rory Kinnear     | 11000.0                | 200074175.0 | Action\|Adventure\|Thriller        | ...  | 994.0                | English  | UK      | PG-13          | 245000000.0 | 2015.0     | 393.0                  | 6.8        | 2.35         | 85000                |
  >|                    The Dark Knight Rises | Color | Christopher Nolan | 813.0                  | 164.0    | 22000.0                 | 23000.0                | Christian Bale   | 27000.0                | 448130642.0 | Action\|Thriller                   | ...  | 2701.0               | English  | USA     | PG-13          | 250000000.0 | 2012.0     | 23000.0                | 8.5        | 2.35         | 164000               |
  >|                                      ... | ...   | ...               | ...                    | ...      | ...                     | ...                    | ...              | ...                    | ...         | ...                                | ...  | ...                  | ...      | ...     | ...            | ...         | ...        | ...                    | ...        | ...          | ...                  |
  >|                            The Following | Color | NaN               | 43.0                   | 43.0     | NaN                     | 319.0                  | Valorie Curry    | 841.0                  | NaN         | Crime\|Drama\|Mystery\|Thriller    | ...  | 359.0                | English  | USA     | TV-14          | NaN         | NaN        | 593.0                  | 7.5        | 16.00        | 32000                |
  >|                     A Plague So Pleasant | Color | Benjamin Roberds  | 13.0                   | 76.0     | 0.0                     | 0.0                    | Maxwell Moody    | 0.0                    | NaN         | Drama\|Horror\|Thriller            | ...  | 3.0                  | English  | USA     | NaN            | 1400.0      | 2013.0     | 0.0                    | 6.3        | NaN          | 16                   |
  >|                         Shanghai Calling | Color | Daniel Hsia       | 14.0                   | 100.0    | 0.0                     | 489.0                  | Daniel Henney    | 946.0                  | 10443.0     | Comedy\|Drama\|Romance             | ...  | 9.0                  | English  | USA     | PG-13          | NaN         | 2012.0     | 719.0                  | 6.3        | 2.35         | 660                  |
  >|                        My Date with Drew | Color | Jon Gunn          | 43.0                   | 90.0     | 16.0                    | 16.0                   | Brian Herzlinger | 86.0                   | 85222.0     | Documentary                        | ...  | 84.0                 | English  | USA     | PG             | 1100.0      | 2004.0     | 23.0                   | 6.6        | 1.85         | 456                  |
  >
  >4916 rows × 27 columns

- 通过reset_index()方法可以重置索引

  ```python
  movie2.reset_index()
  ```

  ><font color = red>输出结果</font>
  >
  >|      |                              movie_title | color |     director_name | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes |     actor_2_name | actor_1_facebook_likes |       gross |  ... | num_user_for_reviews | language | country | content_rating |      budget | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| ---: | ---------------------------------------: | ----: | ----------------: | ---------------------: | -------: | ----------------------: | ---------------------: | ---------------: | ---------------------: | ----------: | ---: | -------------------: | -------: | ------: | -------------: | ----------: | ---------: | ---------------------: | ---------: | -----------: | -------------------- |
  >|    0 |                                   Avatar | Color |     James Cameron |                  723.0 |    178.0 |                     0.0 |                  855.0 | Joel David Moore |                 1000.0 | 760505847.0 |  ... |               3054.0 |  English |     USA |          PG-13 | 237000000.0 |     2009.0 |                  936.0 |        7.9 |         1.78 | 33000                |
  >|    1 | Pirates of the Caribbean: At World's End | Color |    Gore Verbinski |                  302.0 |    169.0 |                   563.0 |                 1000.0 |    Orlando Bloom |                40000.0 | 309404152.0 |  ... |               1238.0 |  English |     USA |          PG-13 | 300000000.0 |     2007.0 |                 5000.0 |        7.1 |         2.35 | 0                    |
  >|    2 |                                  Spectre | Color |        Sam Mendes |                  602.0 |    148.0 |                     0.0 |                  161.0 |     Rory Kinnear |                11000.0 | 200074175.0 |  ... |                994.0 |  English |      UK |          PG-13 | 245000000.0 |     2015.0 |                  393.0 |        6.8 |         2.35 | 85000                |
  >|    3 |                    The Dark Knight Rises | Color | Christopher Nolan |                  813.0 |    164.0 |                 22000.0 |                23000.0 |   Christian Bale |                27000.0 | 448130642.0 |  ... |               2701.0 |  English |     USA |          PG-13 | 250000000.0 |     2012.0 |                23000.0 |        8.5 |         2.35 | 164000               |
  >|  ... |                                      ... |   ... |               ... |                    ... |      ... |                     ... |                    ... |              ... |                    ... |         ... |  ... |                  ... |      ... |     ... |            ... |         ... |        ... |                    ... |        ... |          ... | ...                  |
  >| 4912 |                            The Following | Color |               NaN |                   43.0 |     43.0 |                     NaN |                  319.0 |    Valorie Curry |                  841.0 |         NaN |  ... |                359.0 |  English |     USA |          TV-14 |         NaN |        NaN |                  593.0 |        7.5 |        16.00 | 32000                |
  >| 4913 |                     A Plague So Pleasant | Color |  Benjamin Roberds |                   13.0 |     76.0 |                     0.0 |                    0.0 |    Maxwell Moody |                    0.0 |         NaN |  ... |                  3.0 |  English |     USA |            NaN |      1400.0 |     2013.0 |                    0.0 |        6.3 |          NaN | 16                   |
  >| 4914 |                         Shanghai Calling | Color |       Daniel Hsia |                   14.0 |    100.0 |                     0.0 |                  489.0 |    Daniel Henney |                  946.0 |     10443.0 |  ... |                  9.0 |  English |     USA |          PG-13 |         NaN |     2012.0 |                  719.0 |        6.3 |         2.35 | 660                  |
  >| 4915 |                        My Date with Drew | Color |          Jon Gunn |                   43.0 |     90.0 |                    16.0 |                   16.0 | Brian Herzlinger |                   86.0 |     85222.0 |  ... |                 84.0 |  English |     USA |             PG |      1100.0 |     2004.0 |                   23.0 |        6.6 |         1.85 | 456                  |
  >
  >4916 rows × 28 columns

### 4.2 DataFrame修改行名和列名

- DataFrame创建之后，可以通过rename()方法对原有的行索引名和列名进行修改

  ```python
  movie = pd.read_csv('data/movie.csv', index_col='movie_title')
  movie.index[:5]
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >Index(['Avatar', 'Pirates of the Caribbean: At World's End', 'Spectre',
  >       'The Dark Knight Rises', 'Star Wars: Episode VII - The Force Awakens'],
  >      dtype='object', name='movie_title')
  >```
  ```python
  movie.columns[:5]
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >Index(['color', 'director_name', 'num_critic_for_reviews', 'duration',
  >       'director_facebook_likes'],
  >      dtype='object')
  >```

  ```python
  idx_rename = {'Avatar':'Ratava', 'Spectre': 'Ertceps'} 
  col_rename = {'director_name':'Director Name', 'num_critic_for_reviews': 'Critical Reviews'} 
  movie.rename(index=idx_rename, columns=col_rename).head()
  ```

  ><font color = red>输出结果</font>
  >
  >|                                movie_title | color | Director Name     | Critical Reviews | duration | director_facebook_likes | actor_3_facebook_likes | actor_2_name     | actor_1_facebook_likes | gross       | genres                             | ...  | num_user_for_reviews | language | country | content_rating | budget      | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| -----------------------------------------: | :---- | :---------------- | :--------------- | :------- | :---------------------- | :--------------------- | :--------------- | :--------------------- | :---------- | :--------------------------------- | :--- | :------------------- | :------- | :------ | :------------- | :---------- | :--------- | :--------------------- | :--------- | :----------- | :------------------- |
  >|                                     Ratava | Color | James Cameron     | 723.0            | 178.0    | 0.0                     | 855.0                  | Joel David Moore | 1000.0                 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi | ...  | 3054.0               | English  | USA     | PG-13          | 237000000.0 | 2009.0     | 936.0                  | 7.9        | 1.78         | 33000                |
  >|   Pirates of the Caribbean: At World's End | Color | Gore Verbinski    | 302.0            | 169.0    | 563.0                   | 1000.0                 | Orlando Bloom    | 40000.0                | 309404152.0 | Action\|Adventure\|Fantasy         | ...  | 1238.0               | English  | USA     | PG-13          | 300000000.0 | 2007.0     | 5000.0                 | 7.1        | 2.35         | 0                    |
  >|                                    Ertceps | Color | Sam Mendes        | 602.0            | 148.0    | 0.0                     | 161.0                  | Rory Kinnear     | 11000.0                | 200074175.0 | Action\|Adventure\|Thriller        | ...  | 994.0                | English  | UK      | PG-13          | 245000000.0 | 2015.0     | 393.0                  | 6.8        | 2.35         | 85000                |
  >|                      The Dark Knight Rises | Color | Christopher Nolan | 813.0            | 164.0    | 22000.0                 | 23000.0                | Christian Bale   | 27000.0                | 448130642.0 | Action\|Thriller                   | ...  | 2701.0               | English  | USA     | PG-13          | 250000000.0 | 2012.0     | 23000.0                | 8.5        | 2.35         | 164000               |
  >| Star Wars: Episode VII - The Force Awakens | NaN   | Doug Walker       | NaN              | NaN      | 131.0                   | NaN                    | Rob Walker       | 131.0                  | NaN         | Documentary                        | ...  | NaN                  | NaN      | NaN     | NaN            | NaN         | NaN        | 12.0                   | 7.1        | NaN          | 0                    |
  >
  >5 rows × 27 columns

- 如果不使用rename()，也可以将index 和 columns属性提取出来，修改之后，再赋值回去

  ```python
  movie = pd.read_csv('data/movie.csv', index_col='movie_title')
  index = movie.index
  columns = movie.columns
  index_list = index.tolist()
  column_list = columns.tolist()
  
  index_list[0] = 'Ratava'
  index_list[2] = 'Ertceps'
  column_list[1] = 'Director Name'
  column_list[2] = 'Critical Reviews'
  movie.index = index_list
  movie.columns = column_list
  movie.head()
  ```

  ><font color = red>输出结果</font>
  >
  >|                                            | color |     Director Name | Critical Reviews | duration | director_facebook_likes | actor_3_facebook_likes |     actor_2_name | actor_1_facebook_likes |       gross |                             genres |  ... | num_user_for_reviews | language | country | content_rating |      budget | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes |
  >| -----------------------------------------: | ----: | ----------------: | ---------------: | -------: | ----------------------: | ---------------------: | ---------------: | ---------------------: | ----------: | ---------------------------------: | ---: | -------------------: | -------: | ------: | -------------: | ----------: | ---------: | ---------------------: | ---------: | -----------: | -------------------- |
  >|                                     Ratava | Color |     James Cameron |            723.0 |    178.0 |                     0.0 |                  855.0 | Joel David Moore |                 1000.0 | 760505847.0 | Action\|Adventure\|Fantasy\|Sci-Fi |  ... |               3054.0 |  English |     USA |          PG-13 | 237000000.0 |     2009.0 |                  936.0 |        7.9 |         1.78 | 33000                |
  >|   Pirates of the Caribbean: At World's End | Color |    Gore Verbinski |            302.0 |    169.0 |                   563.0 |                 1000.0 |    Orlando Bloom |                40000.0 | 309404152.0 |         Action\|Adventure\|Fantasy |  ... |               1238.0 |  English |     USA |          PG-13 | 300000000.0 |     2007.0 |                 5000.0 |        7.1 |         2.35 | 0                    |
  >|                                    Ertceps | Color |        Sam Mendes |            602.0 |    148.0 |                     0.0 |                  161.0 |     Rory Kinnear |                11000.0 | 200074175.0 |        Action\|Adventure\|Thriller |  ... |                994.0 |  English |      UK |          PG-13 | 245000000.0 |     2015.0 |                  393.0 |        6.8 |         2.35 | 85000                |
  >|                      The Dark Knight Rises | Color | Christopher Nolan |            813.0 |    164.0 |                 22000.0 |                23000.0 |   Christian Bale |                27000.0 | 448130642.0 |                   Action\|Thriller |  ... |               2701.0 |  English |     USA |          PG-13 | 250000000.0 |     2012.0 |                23000.0 |        8.5 |         2.35 | 164000               |
  >| Star Wars: Episode VII - The Force Awakens |   NaN |       Doug Walker |              NaN |      NaN |                   131.0 |                    NaN |       Rob Walker |                  131.0 |         NaN |                        Documentary |  ... |                  NaN |      NaN |     NaN |            NaN |         NaN |        NaN |                   12.0 |        7.1 |          NaN | 0                    |
  >
  >5 rows × 27 columns

### 4.3 添加、删除、插入列

- 通过dataframe[列名]添加新列

  ```python
  movie = pd.read_csv('data/movie.csv')
  movie['has_seen'] = 0
  # 给新列赋值
  movie['actor_director_facebook_likes'] = (movie['actor_1_facebook_likes'] + 
                                                movie['actor_2_facebook_likes'] + 
                                                movie['actor_3_facebook_likes'] + 
                                                movie['director_facebook_likes'])
  ```

- 调用drop方法删除列

  ```python
  movie = movie.drop('actor_director_facebook_likes', axis='columns')
  ```

- 使用insert()方法插入列 loc 新插入的列在所有列中的位置（0,1,2,3...) column=列名 value=值

  ```python
  movie.insert(loc=0,column='profit',value=movie['gross'] - movie['budget'])
  movie
  ```

  ><font color = red>输出结果</font>
  >
  >|      |      profit | color |     director_name | num_critic_for_reviews | duration | director_facebook_likes | actor_3_facebook_likes |     actor_2_name | actor_1_facebook_likes |       gross |  ... | language | country | content_rating |      budget | title_year | actor_2_facebook_likes | imdb_score | aspect_ratio | movie_facebook_likes | has_seen |
  >| ---: | ----------: | ----: | ----------------: | ---------------------: | -------: | ----------------------: | ---------------------: | ---------------: | ---------------------: | ----------: | ---: | -------: | ------: | -------------: | ----------: | ---------: | ---------------------: | ---------: | -----------: | -------------------: | -------- |
  >|    0 | 523505847.0 | Color |     James Cameron |                  723.0 |    178.0 |                     0.0 |                  855.0 | Joel David Moore |                 1000.0 | 760505847.0 |  ... |  English |     USA |          PG-13 | 237000000.0 |     2009.0 |                  936.0 |        7.9 |         1.78 |                33000 | 0        |
  >|    1 |   9404152.0 | Color |    Gore Verbinski |                  302.0 |    169.0 |                   563.0 |                 1000.0 |    Orlando Bloom |                40000.0 | 309404152.0 |  ... |  English |     USA |          PG-13 | 300000000.0 |     2007.0 |                 5000.0 |        7.1 |         2.35 |                    0 | 0        |
  >|    2 | -44925825.0 | Color |        Sam Mendes |                  602.0 |    148.0 |                     0.0 |                  161.0 |     Rory Kinnear |                11000.0 | 200074175.0 |  ... |  English |      UK |          PG-13 | 245000000.0 |     2015.0 |                  393.0 |        6.8 |         2.35 |                85000 | 0        |
  >|    3 | 198130642.0 | Color | Christopher Nolan |                  813.0 |    164.0 |                 22000.0 |                23000.0 |   Christian Bale |                27000.0 | 448130642.0 |  ... |  English |     USA |          PG-13 | 250000000.0 |     2012.0 |                23000.0 |        8.5 |         2.35 |               164000 | 0        |
  >|  ... |         ... |   ... |               ... |                    ... |      ... |                     ... |                    ... |              ... |                    ... |         ... |  ... |      ... |     ... |            ... |         ... |        ... |                    ... |        ... |          ... |                  ... | ...      |
  >| 4912 |         NaN | Color |               NaN |                   43.0 |     43.0 |                     NaN |                  319.0 |    Valorie Curry |                  841.0 |         NaN |  ... |  English |     USA |          TV-14 |         NaN |        NaN |                  593.0 |        7.5 |        16.00 |                32000 | 0        |
  >| 4913 |         NaN | Color |  Benjamin Roberds |                   13.0 |     76.0 |                     0.0 |                    0.0 |    Maxwell Moody |                    0.0 |         NaN |  ... |  English |     USA |            NaN |      1400.0 |     2013.0 |                    0.0 |        6.3 |          NaN |                   16 | 0        |
  >| 4914 |         NaN | Color |       Daniel Hsia |                   14.0 |    100.0 |                     0.0 |                  489.0 |    Daniel Henney |                  946.0 |     10443.0 |  ... |  English |     USA |          PG-13 |         NaN |     2012.0 |                  719.0 |        6.3 |         2.35 |                  660 | 0        |
  >| 4915 |     84122.0 | Color |          Jon Gunn |                   43.0 |     90.0 |                    16.0 |                   16.0 | Brian Herzlinger |                   86.0 |     85222.0 |  ... |  English |     USA |             PG |      1100.0 |     2004.0 |                   23.0 |        6.6 |         1.85 |                  456 | 0        |
  >
  >4916 rows × 30 columns

## 5 导出和导入数据

### 5.1 pickle文件

- 保存成pickle文件

  - 调用to_pickle方法将以二进制格式保存数据
  - 如要保存的对象是计算的中间结果，或者保存的对象以后会在Python中复用，可把对象保存为.pickle文件
  - 如果保存成pickle文件，只能在python中使用
  - 文件的扩展名可以是.p，.pkl，.pickle

  ```python
  scientists = pd.read_csv('data/scientists.csv')
  scientists.to_pickle('output/scientists_df.pickle')
  ```
  
- 读取pickle文件

  - 可以使用pd.read_pickle函数读取.pickle文件中的数据

  ```python
  scientists_name = pd.read_pickle('output/scientists_df.pickle')
  print(scientists_name)
  ```

  
  

### 5.2 CSV文件

- 保存成CSV文件

  - CSV(逗号分隔值)是很灵活的一种数据存储格式
  - 在CSV文件中，对于每一行，各列采用逗号分隔
  - 除了逗号，还可以使用其他类型的分隔符，比如TSV文件，使用制表符作为分隔符
  - CSV是数据协作和共享的首选格式

  ```python
  names.to_csv('output/scientists_name.csv')
  #设置分隔符为\t
  scientists.to_csv('output/scientists_df.tsv',sep='\t')
  ```

- 不在csv文件中写行名

  ```python
  scientists.to_csv('output/scientists_df_noindex.csv',index=False)
  ```

### 5.3 Excel文件

- 保存成Excel文件

  ```python
  import xlwt
  scientists.to_excel('output/scientists_df.xlsx',sheet_name='scientists',index=False)
  ```
  
- 读取Excel文件

  - 使用pd.read_excel() 读取Excel文件

  ```python
  pd.read_excel('output/scientists_df.xlsx')
  ```

  ><font color = red>输出结果</font>
  >
  >|      |                 Name |       Born |       Died |  Age |         Occupation |
  >| ---: | -------------------: | ---------: | ---------: | ---: | -----------------: |
  >|    0 |    Rosaline Franklin | 1920-07-25 | 1958-04-16 |   37 |            Chemist |
  >|    1 |       William Gosset | 1876-06-13 | 1937-10-16 |   61 |       Statistician |
  >|    2 | Florence Nightingale | 1820-05-12 | 1910-08-13 |   90 |              Nurse |
  >|    3 |          Marie Curie | 1867-11-07 | 1934-07-04 |   66 |            Chemist |
  >|    4 |        Rachel Carson | 1907-05-27 | 1964-04-14 |   56 |          Biologist |
  >|    5 |            John Snow | 1813-03-15 | 1858-06-16 |   45 |          Physician |
  >|    6 |          Alan Turing | 1912-06-23 | 1954-06-07 |   41 | Computer Scientist |
  >|    7 |         Johann Gauss | 1777-04-30 | 1855-02-23 |   77 |      Mathematician |

- 注意 pandas读写excel需要额外安装如下三个包

  ```shell
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xlwt 
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openpyxl
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xlrd 
  ```

  

### 5.4 其它数据格式

- feather文件
  - feather是一种文件格式，用于存储二进制对象
  - feather对象也可以加载到R语言中使用
  - feather格式的主要优点是在Python和R语言之间的读写速度要比CSV文件快
  - feather数据格式通常只用中间数据格式，用于Python和R之间传递数据
  - 一般不用做保存最终数据

<table>
<tr>
  <td>导出方法</td>
  <td>说明</td>
 </tr>
  <tr>
  <td>to_clipboard</td>
  <td>把数据保存到系统剪贴板，方便粘贴</td>
 </tr>
  <tr>
  <td>to_dict</td>
  <td>把数据转换成Python字典</td>
 </tr>
  <tr>
  <td>to_hdf</td>
  <td>把数据保存为HDF格式</td>
 </tr>
  <tr>
  <td>to_html</td>
  <td>把数据转换成HTML</td>
 </tr>
  <tr>
  <td>to_json</td>
  <td>把数据转换成JSON字符串</td>
 </tr>
  <tr>
  <td>to_sql</td>
  <td>把数据保存到SQL数据库</td>
 </tr>
</table>



## 小结

- 创建Series和DataFrame
  - pd.Series
  - pd.DataFrame
- Series常用操作
  - 常用属性
    - index
    - values
    - shape,size,dtype
  - 常用方法
    - max(),min(),std()
    - count()
    - describe()
  - 布尔索引
  - 运算
    - 与数值之间进行算数运算会对每一个元素进行计算
    - 两个Series之间进行计算会索引对齐
- DataFrame常用操作
  - 常用属性
  - 常用方法
  - 布尔索引
  - 运算
- 更改Series和DataFrame
  - 指定行索引名字
    - dataframe.set_index()
    - dataframe.reset_index()
  - 修改行/列名字
    - dataframe.rename(index=,columns = )
    - 获取行/列索引 转换成list之后，修改list再赋值回去
  - 添加、删除、插入列
    - 添加 dataframe['新列‘']
    - 删除 dataframe.drop
    - 插入列 dataframe.insert()
- 导入导出数据
  - pickle
  - csv
  - Excel
  - feather
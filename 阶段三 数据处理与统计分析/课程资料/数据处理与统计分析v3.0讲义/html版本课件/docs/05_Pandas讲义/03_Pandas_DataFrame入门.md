# 3 Pandas DataFrame 入门

## 学习目标

- 掌握DataFrame加载数据文件的方法
- 知道如何加载部分数据
- 知道如何对数据进行简单的分组聚合操作

## 1 加载数据集

- 做数据分析首先要加载数据，并查看其结构和内容，对数据有初步的了解
  - 查看行，列
  - 查看每一列中存储信息的类型

- Pandas 并不是 Python 标准库，所以先导入Pandas

    ```python
    import pandas as pd
    ```

- 导入Pandas库之后，通过read_csv加载文件

  - 加载CSV文件

    - *csv文件 Comma-Separated Values*

    ```python
    nobel = pd.read_csv('data/nobel_prizes.csv', index_col='id') # 加载nobel_prizes.csv文件
    nobel.head() # 展示前5条数据
    ```

    ![image-20220121141321225](img/image-20220121141321225.png)

  - 加载TSV文件

    - tsv文件 Tab-Separated Values

    ```python
    # 参数1 要加载的文件路径，参数2sep 传入分隔符，默认是','  '\t'制表符
    gapminder = pd.read_csv('data/gapminder.tsv',sep='\t')  
    gapminder.head()
    ```

    ![image-20220121141604905](img/image-20220121141604905.png)

- 可以通过Python的内置函数type查看返回的数据类型
  ```
  type(gapminder)
  ```

      ><font color = red>输出结果</font>
      >
      >```shell
      >pandas.core.frame.DataFrame
      >```

- 每个dataframe都有一个shape属性，可以获取DataFrame的行数，列数
    - *注：shape是属性 不是方法 不可以使用df.shape() 会报错*

    ```python
    gapminder.shape
    ```

    ><font color = red>输出结果</font>
    >
    >```shell
    >(1704, 6)
    >```

- 可以通过DataFrame的columns属性获取DataFrame中的列名

      ```python
      gapminder.columns
      ```
        
      ><font color = red>输出结果</font>
      >
      >```shell
      >Index(['country', 'continent', 'year', 'lifeExp', 'pop', 'gdpPercap'], dtype='object')
      >```

- 如何获取每一列的数据类型？

    - 与SQL中的数据表类似，DataFrame中的每一列的数据类型必须相同，不同列的数据类型可以不同
      - 可以通过**dtypes属性**，或者**info()**方法获取数据类型
        
      ```python
        gapminder.dtypes
      ```

      ><font color = red>输出结果</font>
      >
      >```shell
      >country       object
      >continent     object
      >year           int64
      >lifeExp      float64
      >pop            int64
      >gdpPercap    float64
      >dtype: object
      >```
      ```python
      gapminder.info()
      ```

      ><font color = red>输出结果</font>
      >
      >```shell
      ><class 'pandas.core.frame.DataFrame'>
      >RangeIndex: 1704 entries, 0 to 1703
      >Data columns (total 6 columns):
      >#   Column     Non-Null Count  Dtype  
      >---  ------     --------------  -----  
      >0   country    1704 non-null   object 
      >1   continent  1704 non-null   object 
      >2   year       1704 non-null   int64  
      >3   lifeExp    1704 non-null   float64
      >4   pop        1704 non-null   int64  
      >5   gdpPercap  1704 non-null   float64
      >dtypes: float64(2), int64(2), object(2)
      >memory usage: 80.0+ KB
      >```

- Pandas与Python常用数据类型对照
    <table>
      <tr>
        <td>Pandas类型</td><td>Python类型</td><td>说明</td>
      </tr>
      <tr>
        <td>Object</td><td>string</td><td>字符串类型</td>
      </tr>
      <tr>
        <td>int64</td><td>int</td><td>整形</td>
      </tr>
      <tr>
        <td>float64</td><td>float</td><td>浮点型</td>
      </tr>
      <tr>
        <td>datetime64</td><td>datetime</td><td>日期时间类型，python中需要加载</td>
      </tr>
    </table>

## 2 查看部分数据

### 2.1 根据列名加载部分列数据

- 加载一列数据，通过DataFrame['列名']方式获取

  ```python
  nobel_firstname = nobel['firstname']
  #获取数据前5行
  nobel_firstname.head()
  ```
  
  ><font color = red>输出结果</font>
  >
  >```shell
  >id
  >941      Rainer
  >942    Barry C.
  >943      Kip S.
  >944     Jacques
  >945     Joachim
  >Name: firstname, dtype: object
  >```
  
- 通过列名加载多列数据，通过DataFrame[['列名1','列名2',...]]

  - 注意这里是两层[] 可以理解为DataFrame[列名的list]

  ```python
  subset = nobel[['firstname', 'surname']]
  #打印后五行数据
  subset.tail()
  ```

  > <font color = red>输出结果</font>
  >
  >```shell
  >    firstname	surname
  >id		
  >160	Jacobus Henricus	van 't Hoff
  >293	Emil Adolf	von Behring
  >569	Sully	Prudhomme
  >462	Jean Henry	Dunant
  >463	Frédéric	Passy
  >```

### 2.2 按行加载部分数据

- loc：通过行索引标签获取指定行数据

  ```python
  #先打印前5行数据 观察第一列
  nobel.head()
  ```

  <font color = red>显示结果</font>

  ```shell
      year	category	overallMotivation	firstname	surname	motivation	share
  id							
  941	2017	physics	NaN	Rainer	Weiss	"for decisive contributions to the LIGO detect...	2
  942	2017	physics	NaN	Barry C.	Barish	"for decisive contributions to the LIGO detect...	4
  943	2017	physics	NaN	Kip S.	Thorne	"for decisive contributions to the LIGO detect...	4
  944	2017	chemistry	NaN	Jacques	Dubochet	"for developing cryo-electron microscopy for t...	3
  945	2017	chemistry	NaN	Joachim	Frank	"for developing cryo-electron microscopy for t...	3
  ```

  - 上述结果中发现，最左边一列是id，这一列的数据是DataFrame的**行索引**，默认情况Pandas会使用[0,1,2,...]作为的行索引，这里我们指定了id作为行索引。
  - 我们可以使用 .loc 方法传入行索引，来获取DataFrame的部分数据（一行，或多行）

  ```python
  # 获取第一行数据
  nobel.loc[941]
  ```

  <font color = red>显示结果</font>

  ```shell
  year                                                              2017
  category                                                       physics
  overallMotivation                                                  NaN
  firstname                                                       Rainer
  surname                                                          Weiss
  motivation           "for decisive contributions to the LIGO detect...
  share                                                                2
  Name: 941, dtype: object
  ```

  - 使用tail方法获取最后一行数据

    ```python
    df.tail(n=1) #tail方法默认输出一行 传入n=1控制只显示1行
    ```

    > <font color = red>输出结果</font>
    >
    > ```shell
    >     year	category	overallMotivation	firstname	surname	motivation	share
    > id							
    > 463	1901	peace	NaN	Frédéric	Passy	NaN	2
    > ```

    ```python
    # 获取最后一行数据，并打印
    nobel.loc[463]
    ```

    <font color = red>显示结果</font>

    ```shell
    year                     1901
    category                peace
    overallMotivation         NaN
    firstname            Frédéric
    surname                 Passy
    motivation                NaN
    share                       2
    Name: 463, dtype: object
    ```

  - 注意：nobel.loc 和 nobel.tail 两种方式获得的最后一行数据有些不同
    - 我们可以打印两种结果的类型

    ```python
    subset_loc = nobel.loc[463] 
    subset_tail = nobel.tail(n=1) 
    print(type(subset_loc))
    print(type(subset_head))
    ```

    ><font color = red>输出结果</font>
    >
    >```shell
    ><class 'pandas.core.series.Series'>
    ><class ’pandas.core.frame.DataFrame’>
    >```

- loc：通过索引标签获取指定多行数据

  ```python
  nobel.loc[[941, 945]]
  ```

  > <font color = red>输出结果</font>
  >
  > ```shell
  >        year	category	overallMotivation	firstname	surname	motivation	share
  > id							
  > 941	2017	physics	NaN	Rainer	Weiss	"for decisive contributions to the LIGO detect...	2
  > 945	2017	chemistry	NaN	Joachim	Frank	"for developing cryo-electron microscopy for t...	3
  > ```

- iloc : 通过行号获取行数据

  - 在当前案例中，使用iloc 和 loc效果是一样的
  - 需要注意的是，iloc传入的是索引的序号，loc是索引的标签
    - 在当前案例中，索引标签和索引序号刚好相同
    - 并不是所有情况下索引标签=索引序号
    - 例如：在做时间序列分析的时候，我们可以使用日期作为行索引
      - 此时索引标签日期
      - 索引序号依然是0，1，2，3

  ```python
  # 获取第一行数据，并打印
  nobel.iloc[0]
  ```

  > <font color = red>输出结果</font>
  >
  > ```
  > year                                                              2017
  > category                                                       physics
  > overallMotivation                                                  NaN
  > firstname                                                       Rainer
  > surname                                                          Weiss
  > motivation           "for decisive contributions to the LIGO detect...
  > share                                                                2
  > Name: 941, dtype: object
  > ```
  
  ```python
  #获取第100行数据，并打印
  nobel.iloc[99]
  ```
  
  > <font color = red>输出结果</font>
  >
  > ```
  > year                                                              2009
  > category                                                      medicine
  > overallMotivation                                                  NaN
  > firstname                                                 Elizabeth H.
  > surname                                                      Blackburn
  > motivation           "for the discovery of how chromosomes are prot...
  > share                                                                3
  > Name: 835, dtype: object
  > ```
  
  ```python
  # 获取最后一行  通过shape 获取一共有多少行
  number_of_rows = nobel.shape[0]
  # 总行数-1 获取最后一行行索引
  last_row_index = number_of_rows - 1 
  # 获取最后一行数据，并打印
  nobel.iloc[last_row_index]
  ```
  
  > <font color = red>输出结果</font>
  >
  > ```
  > year                     1901
  > category                peace
  > overallMotivation         NaN
  > firstname            Frédéric
  > surname                 Passy
  > motivation                NaN
  > share                       2
  > Name: 463, dtype: object
  > ```
  
  - 使用iloc时传入-1可以获取最后一行数据
  
  ```python
  nobel.iloc[-1]
  ```
  
  > <font color = red>输出结果</font>
  >
  > ```
  > year                     1901
  > category                peace
  > overallMotivation         NaN
  > firstname            Frédéric
  > surname                 Passy
  > motivation                NaN
  > share                       2
  > Name: 463, dtype: object
  > ```
  
  - **注意** 使用iloc时可以传入-1来获取最后一行数据，使用loc的时候不行 


### 2.3 获取指定行/列数据

- loc和iloc属性既可以用于获取列数据，也可以用于获取行数据

  - df.loc[[行]，[列]]
  - df.iloc[[行]，[列]]

- 使用 loc 获取数据中的1列/几列

  - df.loc[[所有行],[列名]] 
  - 取出所有行，可以使用切片语法 df.loc[ : , [列名]]

  ```python
  subset = nobel.loc[:, ['firstname', 'surname']]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >    firstname	surname
  >id		
  >941	Rainer	Weiss
  >942	Barry C.	Barish
  >943	Kip S.	Thorne
  >944	Jacques	Dubochet
  >945	Joachim	Frank
  >```

- 使用 iloc 获取数据中的1列/几列

  - df.iloc[:,[列序号]]  # 列序号可以使用-1代表最后一列

  ```python
  subset = nobel.iloc[:,[1,4,-1]]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```
  >    category	surname	share
  >id			
  >941	physics	Weiss	2
  >942	physics	Barish	4
  >943	physics	Thorne	4
  >944	chemistry	Dubochet	3
  >945	chemistry	Frank	3
  >```

- 如果loc 和 iloc 传入的参数弄混了，会报错

  - loc 只能接受行/列 的名字， 不能传入索引

  ```python
  subset = nobel.loc[:,[1,4,-1]]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >KeyError: "None of [Int64Index([1, 4, -1], dtype='int64')] are in the [columns]"
  >```
  - iloc只能接受行/列的索引，不能传入行名，或者列名

  ```python
  subset = nobel.iloc[:,['category','surname','share']]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >IndexError: .iloc requires numeric indexers, got ['category' 'surname' 'share']
  >```

- 通过range 生成序号，结合iloc 获取连续多列数据

  ```python
  tmp_range = list(range(5))
  tmp_range
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >[0, 1, 2, 3, 4]
  >```

  ```python
  subset = df.iloc[:,tmp_range]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >    year	category	overallMotivation	firstname	surname
  >id					
  >941	2017	physics	NaN	Rainer	Weiss
  >942	2017	physics	NaN	Barry C.	Barish
  >943	2017	physics	NaN	Kip S.	Thorne
  >944	2017	chemistry	NaN	Jacques	Dubochet
  >945	2017	chemistry	NaN	Joachim	Frank
  >```

- 在 iloc中使用切片语法获取几列数据

  - 使用切片语法获取前三列

  ```python
  subset = nobel.iloc[:,3:6]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >	firstname	surname	motivation
  >id			
  >941	Rainer	Weiss	"for decisive contributions to the LIGO detect...
  >942	Barry C.	Barish	"for decisive contributions to the LIGO detect...
  >943	Kip S.	Thorne	"for decisive contributions to the LIGO detect...
  >944	Jacques	Dubochet	"for developing cryo-electron microscopy for t...
  >945	Joachim	Frank	"for developing cryo-electron microscopy for t...
  >```

  - 获取第0,2,4列

  ```python
  subset = nobel.iloc[:,0:6:2]
  subset.head()
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >    year	overallMotivation	surname
  >id			
  >941	2017	NaN	Weiss
  >942	2017	NaN	Barish
  >943	2017	NaN	Thorne
  >944	2017	NaN	Dubochet
  >945	2017	NaN	Frank
  >```

- 使用 loc/iloc 获取指定行，指定列的数据

  - 使用loc

  ```python
  nobel.loc[941,'firstname']
  ```

  ><font color = red>输出结果</font>
  >
  >```
  >Rainer
  >```
  - 使用iloc

  ```python
  nobel.iloc[0,3]
  ```

  ><font color = red>输出结果</font>
  >
  >```
  >Rainer
  >```

  - 不要混淆loc和iloc，df.loc[42,0] 会报错

  ```python
  nobel.loc[0,3]
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >KeyError: 0
  >```

- 获取多行多列

  - 可以把获取单行单列的语法和获取多行多列的语法结合起来使用
  - 获取 第一列，第四列，第六列（country,lifeExp,gdpPercap) 数据中的第1行，第20行和第30行

  ```python
  nobel.iloc[[0,19,29],[0,3,5]]
  ```

  ><font color = red>输出结果</font>
  >
  >```shell
  >	year	firstname	motivation
  >id			
  >941	2017	Rainer	"for decisive contributions to the LIGO detect...
  >937	2016	Bob	"for having created new poetic expressions wit...
  >917	2015	Satoshi	"for their discoveries concerning a novel ther...
  >```
  - 在实际工作中，获取某几列数据的时候，建议传入实际的列名，使用列名的好处：
    - 增加代码的可读性
    - 避免因列顺序的变化导致取出错误的列数据
  
  ```python
  nobel.loc[[941,937,917],['year','firstname','motivation']]
  ```
  
  ><font color = red>输出结果</font>
  >
  >```shell
  >	year	firstname	motivation
  >id			
  >941	2017	Rainer	"for decisive contributions to the LIGO detect...
  >937	2016	Bob	"for having created new poetic expressions wit...
  >917	2015	Satoshi	"for their discoveries concerning a novel ther...
  >```

  - 注意：可以在loc 和 iloc 属性的行部分使用切片获取数据
  
  ```python
  nobel.loc[941:938,['year','firstname','motivation']]
  ```
  
  ><font color = red>输出结果</font>
  >
  >```shell
  >	year	firstname	motivation
  >id			
  >941	2017	Rainer	"for decisive contributions to the LIGO detect...
  >942	2017	Barry C.	"for decisive contributions to the LIGO detect...
  >943	2017	Kip S.	"for decisive contributions to the LIGO detect...
  >944	2017	Jacques	"for developing cryo-electron microscopy for t...
  >945	2017	Joachim	"for developing cryo-electron microscopy for t...
  >946	2017	Richard	"for developing cryo-electron microscopy for t...
  >938	2017	Jeffrey C.	"for their discoveries of molecular mechanisms...
  >```

## 3 分组和聚合计算

- 在我们使用Excel或者SQL进行数据处理时，Excel和SQL都提供了基本的统计计算功能

- 当我们再次查看gapminder数据的时候，可以根据数据提出几个问题

  ```python
  print(gapminder.head(10))
  ```

  > <font color = red>输出结果</font>
  >
  >   ```shell
  >        country continent  year  lifeExp       pop   gdpPercap
  > 0  Afghanistan      Asia  1952   28.801   8425333  779.445314
  > 1  Afghanistan      Asia  1957   30.332   9240934  820.853030
  > 2  Afghanistan      Asia  1962   31.997  10267083  853.100710
  > 3  Afghanistan      Asia  1967   34.020  11537966  836.197138
  > 4  Afghanistan      Asia  1972   36.088  13079460  739.981106
  > 5  Afghanistan      Asia  1977   38.438  14880372  786.113360
  > 6  Afghanistan      Asia  1982   39.854  12881816  978.011439
  > 7  Afghanistan      Asia  1987   40.822  13867957  852.395945
  > 8  Afghanistan      Asia  1992   41.674  16317921  649.341395
  > 9  Afghanistan      Asia  1997   41.763  22227415  635.341351
  >   ```

  ① 每一年的平均预期寿命是多少？每一年的平均人口和平均GDP是多少？

  ② 如果我们按照大洲来计算，每年每个大洲的平均预期寿命，平均人口，平均GDP情况又如何？

  ③ 在数据中，每个大洲列出了多少个国家和地区？

### 3.1 分组方式

- 对于上面提出的问题，需要进行分组-聚合计算

  - 先将数据分组（每一年的平均预期寿命问题 按照年份将相同年份的数据分成一组） 
  - 对每组的数据再去进行统计计算如，求平均，求每组数据条目数（频数）等
  - 再将每一组计算的结果合并起来
  - 可以使用DataFrame的groupby方法完成分组/聚合计算

  ```python
  print(gapminder.groupby('year')['lifeExp'].mean())
  ```

  > <font color = red>显示结果</font>
  >
  > ```shell
  > year
  > 1952    49.057620
  > 1957    51.507401
  > 1962    53.609249
  > 1967    55.678290
  > 1972    57.647386
  > 1977    59.570157
  > 1982    61.533197
  > 1987    63.212613
  > 1992    64.160338
  > 1997    65.014676
  > 2002    65.694923
  > 2007    67.007423
  > Name: lifeExp, dtype: float64
  > ```

- 我们将上面一行代码拆开，逐步分析

  - 通过gapminder.groupby('year')先创一个分组对象，如果打印这个分组的DataFrame，会返回一个内存地址

  ```python
  grouped_year_gapminder = gapminder.groupby('year')
  print(type(grouped_year_gapminder))
  print(grouped_year_gapminder)
  ```

  ><font color = red>显示结果</font>
  >
  >```shell
  ><class 'pandas.core.groupby.generic.DataFrameGroupBy'>
  ><pandas.core.groupby.generic.DataFrameGroupBy object at 0x123493f10>
  >```

  - 我们可以从分组之后的数据DataFrameGroupBy中，传入列名获取我们感兴趣的数据，并进行进一步计算
    - 计算每一年的平均预期寿命，我们需要用到 *lifeExp*  这一列
    - 我们可以使用上一小节介绍的方法获取分组之后数据中的一列

  ```python
  grouped_year_gapminder_lifeExp = grouped_year_gapminder['lifeExp']
  print(type(grouped_year_gapminder_lifeExp)) 
  print(grouped_year_gapminder_lifeExp)
  ```

  ><font color = red>显示结果</font>
  >
  >```shell
  ><class 'pandas.core.groupby.generic.SeriesGroupBy'>
  ><pandas.core.groupby.generic.SeriesGroupBy object at 0x000001E1938D0710>
  >```

  - 返回结果为一个 SeriesGroupBy （只获取了DataFrameGroupBy中的一列），其内容是分组后的数据
  - 对分组后的数据计算平均值

  ```python
  mean_lifeExp_by_year = grouped_year_gapminder_lifeExp.mean()
  print(mean_lifeExp_by_year)
  ```

  ><font color = red>显示结果</font>
  >
  >```shell
  >year
  >1952    49.057620
  >1957    51.507401
  >1962    53.609249
  >1967    55.678290
  >1972    57.647386
  >1977    59.570157
  >1982    61.533197
  >1987    63.212613
  >1992    64.160338
  >1997    65.014676
  >2002    65.694923
  >2007    67.007423
  >Name: lifeExp, dtype: float64
  >```

- 上面的例子只是对一列 *lifeExp* 进行了分组求平均，如果想对多列值进行分组聚合代码也类似

  ```python
  print(gapminder.groupby(['year', 'continent'])[['lifeExp','gdpPercap']].mean())
  ```

  ><font color = red>显示结果</font>
  >
  >```shell
  >                  lifeExp     gdpPercap
  >year continent                         
  >1952 Africa     39.135500   1252.572466
  >     Americas   53.279840   4079.062552
  >     Asia       46.314394   5195.484004
  >     Europe     64.408500   5661.057435
  >     Oceania    69.255000  10298.085650
  >1957 Africa     41.266346   1385.236062
  >     Americas   55.960280   4616.043733
  >     Asia       49.318544   5787.732940
  >     Europe     66.703067   6963.012816
  >     Oceania    70.295000  11598.522455
  >1962 Africa     43.319442   1598.078825
  >     Americas   58.398760   4901.541870
  >     Asia       51.563223   5729.369625
  >     Europe     68.539233   8365.486814
  >     Oceania    71.085000  12696.452430
  >1967 Africa     45.334538   2050.363801
  >     Americas   60.410920   5668.253496
  >     Asia       54.663640   5971.173374
  >     Europe     69.737600  10143.823757
  >     Oceania    71.310000  14495.021790
  >1972 Africa     47.450942   2339.615674
  >     Americas   62.394920   6491.334139
  >     Asia       57.319269   8187.468699
  >     Europe     70.775033  12479.575246
  >     Oceania    71.910000  16417.333380
  >1977 Africa     49.580423   2585.938508
  >     Americas   64.391560   7352.007126
  >     Asia       59.610556   7791.314020
  >     Europe     71.937767  14283.979110
  >     Oceania    72.855000  17283.957605
  >1982 Africa     51.592865   2481.592960
  >     Americas   66.228840   7506.737088
  >     Asia       62.617939   7434.135157
  >     Europe     72.806400  15617.896551
  >     Oceania    74.290000  18554.709840
  >1987 Africa     53.344788   2282.668991
  >     Americas   68.090720   7793.400261
  >     Asia       64.851182   7608.226508
  >     Europe     73.642167  17214.310727
  >     Oceania    75.320000  20448.040160
  >1992 Africa     53.629577   2281.810333
  >     Americas   69.568360   8044.934406
  >     Asia       66.537212   8639.690248
  >     Europe     74.440100  17061.568084
  >     Oceania    76.945000  20894.045885
  >1997 Africa     53.598269   2378.759555
  >     Americas   71.150480   8889.300863
  >     Asia       68.020515   9834.093295
  >     Europe     75.505167  19076.781802
  >     Oceania    78.190000  24024.175170
  >2002 Africa     53.325231   2599.385159
  >     Americas   72.422040   9287.677107
  >     Asia       69.233879  10174.090397
  >     Europe     76.700600  21711.732422
  >     Oceania    79.740000  26938.778040
  >2007 Africa     54.806038   3089.032605
  >     Americas   73.608120  11003.031625
  >     Asia       70.728485  12473.026870
  >     Europe     77.648600  25054.481636
  >     Oceania    80.719500  29810.188275
  >```
  - 上面的代码按年份和大洲对数据进行分组，针对每一组数据计算了对应的平均预期寿命 *lifeExp* 和 平均GDP
  - 输出的结果中  year continent 和 lifeExp gdpPercap 不在同一行， year continent两个行索引存在层级结构，后面的章节会详细介绍这种复合索引的用法
  - 如果想去掉 year continent的层级结构，可以使用reset_index方法（重置行索引）

  ```python
  multi_group_var = gapminder.groupby(['year', 'continent'])[['lifeExp','gdpPercap']].mean()
  flat = multi_group_var.reset_index()
  print(flat.head(15))
  ```

  ><font color = red>显示结果</font>
  >
  >```python
  >    year continent    lifeExp     gdpPercap
  >0   1952    Africa  39.135500   1252.572466
  >1   1952  Americas  53.279840   4079.062552
  >2   1952      Asia  46.314394   5195.484004
  >3   1952    Europe  64.408500   5661.057435
  >4   1952   Oceania  69.255000  10298.085650
  >5   1957    Africa  41.266346   1385.236062
  >6   1957  Americas  55.960280   4616.043733
  >7   1957      Asia  49.318544   5787.732940
  >8   1957    Europe  66.703067   6963.012816
  >9   1957   Oceania  70.295000  11598.522455
  >10  1962    Africa  43.319442   1598.078825
  >11  1962  Americas  58.398760   4901.541870
  >12  1962      Asia  51.563223   5729.369625
  >13  1962    Europe  68.539233   8365.486814
  >14  1962   Oceania  71.085000  12696.452430
  >```

### 3.2 分组频数计算

- 在数据分析中，一个常见的任务是计算频数

  - 可以使用 *nunique* 方法 计算Pandas Series的唯一值计数
  - 可以使用 *value_counts* 方法来获取Pandas Series 的频数统计
  - 在数据中，每个大洲列出了多少个国家和地区？

  ```python
  gapminder.groupby('continent')['country'].nunique()
  ```

  ><font color = red>显示结果</font>
  >
  >```
  >continent
  >Africa      52
  >Americas    25
  >Asia        33
  >Europe      30
  >Oceania      2
  >Name: country, dtype: int64
  >```

## 4 简单绘图

- 可视化在数据分析的每个步骤中都非常重要，在理解或清理数据时，可视化有助于识别数据中的趋势

  ```python
  global_yearly_life_expectancy = gapminder.groupby('year')['lifeExp'].mean()
  print(global_yearly_life_expectancy)
  ```

  > <font color = red>显示结果</font>
  >
  > ```shell
  > year
  > 1952    49.057620
  > 1957    51.507401
  > 1962    53.609249
  > 1967    55.678290
  > 1972    57.647386
  > 1977    59.570157
  > 1982    61.533197
  > 1987    63.212613
  > 1992    64.160338
  > 1997    65.014676
  > 2002    65.694923
  > 2007    67.007423
  > Name: lifeExp, dtype: float64
  > ```
  - 可以通过plot画图

  ``` python
  global_yearly_life_expectancy.plot()
  ```

  > <font color = red>显示结果</font>
  >
  > ```shell
  > <matplotlib.axes._subplots.AxesSubplot at 0x1e196e73f98>
  > ```
  >
  > ![img](data:image/png;base64,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)



## 小结

- 本节课程介绍了如何使用Pandas的DataFrame加载数据，并介绍了如何对数据进行简单的分组聚合

  ```python
  pd.read_csv # 加载CSV文件
  pd.loc      # 从DataFrame中获取部分数据，传入索引名字
  pd.iloc     # 从DataFrame中获取部分数据，传入索引序号
  pd.groupby  # 分组
  ```

  


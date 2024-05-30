# 7 apply 自定义函数

## 学习目标

- 掌握apply的用法
- 知道如何创建向量化函数

## 1 简介

- Pandas提供了很多数据处理的API,但当提供的API不能满足需求的时候,需要自己编写数据处理函数, 这个时候可以使用apply函数
- apply函数可以接收一个自定义函数, 可以将DataFrame的行/列数据传递给自定义函数处理
- apply函数类似于编写一个for循环, 遍历行/列的每一个元素,但比使用for循环效率高很多

## 2 Series的apply方法

- 数据准备

```python
import pandas as pd
df = pd.DataFrame({'a':[10,20,30],'b':[20,30,40]})
df
```

><font color = 'red'>显示结果:</font>
>
>|      |    a |    b |
>| ---: | ---: | ---: |
>|    0 |   10 |   20 |
>|    1 |   20 |   30 |
>|    2 |   30 |   40 |

- 创建一个自定义函数

```python
def my_sq(x):
    """
    求平方
    """
    return x**2
```

- Series有一个apply方法, 该方法有一个func参数,当传入一个函数后,apply方法就会把传入的函数应用于Series的每个元素

```python
sq = df['a'].apply(my_sq)
sq
```

><font color='red'>显示结果:</font>
>
>```shell
>0    100
>1    400
>2    900
>Name: a, dtype: int64
>```

- 注意,把my_sq传递给apply的时候,不要加上圆括号

```python
def my_exp(x,e):
    return x**e
my_exp(2,3)
```

><font color='red'>显示结果:</font>
>
>```
>8
>```

- apply 传入 需要多个参数的函数

```python
ex = df['a'].apply(my_exp,e=2)
ex
```

><font color='red'>显示结果:</font>
>
>```shell
>0    100
>1    400
>2    900
>Name: a, dtype: int64
>```

## 3 DataFrame的apply方法

- 把上面创建的my_sq, 直接应用到整个DataFrame中

```python
df.apply(my_sq)
```

><font color='red'>显示结果:</font>
>
>|      |    a |    b |
>| ---: | ---: | ---: |
>|    0 |  100 |  400 |
>|    1 |  400 |  900 |
>|    2 |  900 | 1600 |

- dataframe是二维数据, 

```python
# 编写函数计算列的平均值
def avg_3(x,y,z):
    return (x+y+z)/3
df.apply(avg_3)
```

><font color='red'>显示结果:</font>
>
>```shell
>---------------------------------------------------------------------------
>TypeError                                 Traceback (most recent call last)
><ipython-input-39-3a016d8b8964> in <module>
>      1 def avg_3(x,y,z):
>      2     return (x+y+z)/3
>----> 3 df.apply(avg_3)
>
>~\anaconda3\lib\site-packages\pandas\core\frame.py in apply(self, func, axis, raw, result_type, args, **kwds)
>   6876             kwds=kwds,
>   6877         )
>-> 6878         return op.get_result()
>   6879 
>   6880     def applymap(self, func) -> "DataFrame":
>
>~\anaconda3\lib\site-packages\pandas\core\apply.py in get_result(self)
>    184             return self.apply_raw()
>    185 
>--> 186         return self.apply_standard()
>    187 
>    188     def apply_empty_result(self):
>
>~\anaconda3\lib\site-packages\pandas\core\apply.py in apply_standard(self)
>    294             try:
>    295                 result = libreduction.compute_reduction(
>--> 296                     values, self.f, axis=self.axis, dummy=dummy, labels=labels
>    297                 )
>    298             except ValueError as err:
>
>pandas\_libs\reduction.pyx in pandas._libs.reduction.compute_reduction()
>
>pandas\_libs\reduction.pyx in pandas._libs.reduction.Reducer.get_result()
>
>TypeError: avg_3() missing 2 required positional arguments: 'y' and 'z'
>```

- 从报错的信息中看到,实际上传入avg_3函数中的只有一个变量,这个变量可以是DataFrame的行也可以是DataFrame的列, 使用apply的时候,可以通过axis参数指定按行/ 按列 传入数据
  - axis = 0 (默认) 按列处理
  - axis = 1  按行处理

```python
# 修改avg_3函数
def avg_3_apply(col):
    x = col[0]
    y = col[1]
    z = col[2]
    return (x+y+z)/3
df.apply(avg_3_apply)
```

><font color='red'>显示结果:</font>
>
>```shell
>a    20.0
>b    30.0
>dtype: float64
>```



```python
#按行处理
def avg_2_apply(row):
    x = row[0]
    y = row[1]
    return (x+y)/2
df.apply(avg_2_apply,axis=1)
```

><font color='red'>显示结果:</font>
>
>```shell
>0    15.0
>1    25.0
>2    35.0
>dtype: float64
>```

```python
# 通用写法
def avg_apply(s):
    return s.mean()
```



## 4 apply 使用案例

- 接下来使用titanic数据集来介绍apply的用法

```python
#加载数据,使用info查看该数据集的基本特征
titanic = pd.read_csv('data/titanic.csv')
titanic.info()
```

><font color='red'>显示结果:</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 891 entries, 0 to 890
>Data columns (total 15 columns):
> #   Column       Non-Null Count  Dtype  
>---  ------       --------------  -----  
> 0   survived     891 non-null    int64  
> 1   pclass       891 non-null    int64  
> 2   sex          891 non-null    object 
> 3   age          714 non-null    float64
> 4   sibsp        891 non-null    int64  
> 5   parch        891 non-null    int64  
> 6   fare         891 non-null    float64
> 7   embarked     889 non-null    object 
> 8   class        891 non-null    object 
> 9   who          891 non-null    object 
> 10  adult_male   891 non-null    bool   
> 11  deck         203 non-null    object 
> 12  embark_town  889 non-null    object 
> 13  alive        891 non-null    object 
> 14  alone        891 non-null    bool   
>dtypes: bool(2), float64(2), int64(4), object(7)
>memory usage: 92.4+ KB
>```

- 该数据集有891行,15列, 其中age 和 deck 两列中包含缺失值.可以使用apply计算数据中有多少null 或 NaN值

  - 缺失值数目

  ```python
  import numpy as np
  def count_missing(vec):
      """
      计算一个向量中缺失值的个数
      """
      #根据值是否缺失获取一个由True/False组成的向量
      null_vec = pd.isnull(vec)
      # 得到null_vec中null值的个数
      # null值对应True, True为1
      null_count = np.sum(null_vec)
      #返回向量中缺失值的个数
      return null_count
  ```

  - 缺失值占比

  ```python
  def prop_missing(vec):
      """
      向量中缺失值的占比
      """
      # 计算缺失值的个数
      # 这里使用刚刚编写的count_missing函数
      num = count_missing(vec)
      #获得向量中元素的个数
      #也需要统计缺失值个数
      dem = vec.size
      return num/dem
  ```

  - 非缺失值占比

  ```python
  def prop_complete(vec):
      """
      向量中非缺失值(完整值)的占比
      """
      #先计算缺失值占的比例
      #然后用1减去缺失值的占比
      return 1-prop_missing(vec)
  ```

- 把前面定义好的函数应用于数据的各列

  ```python
  titanic.apply(count_missing)
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >survived         0
  >pclass           0
  >sex              0
  >age            177
  >sibsp            0
  >parch            0
  >fare             0
  >embarked         2
  >class            0
  >who              0
  >adult_male       0
  >deck           688
  >embark_town      2
  >alive            0
  >alone            0
  >dtype: int64
  >```

  ```python
  titanic.apply(prop_missing)
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >survived       0.000000
  >pclass         0.000000
  >sex            0.000000
  >age            0.198653
  >sibsp          0.000000
  >parch          0.000000
  >fare           0.000000
  >embarked       0.002245
  >class          0.000000
  >who            0.000000
  >adult_male     0.000000
  >deck           0.772166
  >embark_town    0.002245
  >alive          0.000000
  >alone          0.000000
  >dtype: float64
  >```

  ```python
  titanic.apply(prop_complete)
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >survived       1.000000
  >pclass         1.000000
  >sex            1.000000
  >age            0.801347
  >sibsp          1.000000
  >parch          1.000000
  >fare           1.000000
  >embarked       0.997755
  >class          1.000000
  >who            1.000000
  >adult_male     1.000000
  >deck           0.227834
  >embark_town    0.997755
  >alive          1.000000
  >alone          1.000000
  >dtype: float64
  >```

- 按行使用

  ```python
  titanic.apply(count_missing,axis = 1)
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >0      1
  >1      0
  >2      1
  >3      0
  >4      1
  >      ..
  >886    1
  >887    0
  >888    2
  >889    0
  >890    1
  >Length: 891, dtype: int64
  >```

  ```python
  titanic.apply(prop_missing,axis = 1)
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >0      0.066667
  >1      0.000000
  >2      0.066667
  >3      0.000000
  >4      0.066667
  >         ...   
  >886    0.066667
  >887    0.000000
  >888    0.133333
  >889    0.000000
  >890    0.066667
  >Length: 891, dtype: float64
  >```

  ```python
  titanic.apply(prop_complete,axis = 1)
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >0      0.933333
  >1      1.000000
  >2      0.933333
  >3      1.000000
  >4      0.933333
  >         ...   
  >886    0.933333
  >887    1.000000
  >888    0.866667
  >889    1.000000
  >890    0.933333
  >Length: 891, dtype: float64
  >```

  ```python
  titanic.apply(count_missing,axis = 1).value_counts()
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >1    549
  >0    182
  >2    160
  >dtype: int64
  >```

## 5 向量化函数

- 创建一个DataFrame

  ```python
  df = pd.DataFrame({'a':[10,20,30],'b':[20,30,40]})
  df
  ```

  ><font color='red'>显示结果:</font>
  >
  >|      |    a |    b |
  >| ---: | ---: | ---: |
  >|    0 |   10 |   20 |
  >|    1 |   20 |   30 |
  >|    2 |   30 |   40 |

- 创建函数

  ```python
  def avg_2_mod(x,y):
      if(x==20):
          return (np.NaN)
      else:
          return (x+y)/2
  avg_2_mod(df['a'],df['b'])
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >---------------------------------------------------------------------------
  >ValueError                                Traceback (most recent call last)
  ><ipython-input-71-760ddffd820b> in <module>
  >----> 1 avg_2_mod(df['a'],df['b'])
  >
  ><ipython-input-70-ec94cdbae859> in avg_2_mod(x, y)
  >      1 def avg_2_mod(x,y):
  >----> 2     if(x==20):
  >      3         return (np.NaN)
  >      4     else:
  >      5         return (x+y)/2
  >
  >~\anaconda3\lib\site-packages\pandas\core\generic.py in __nonzero__(self)
  >   1477     def __nonzero__(self):
  >   1478         raise ValueError(
  >-> 1479             f"The truth value of a {type(self).__name__} is ambiguous. "
  >   1480             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
  >   1481         )
  >
  >ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
  >```

- 上面函数中, x==20 , x 是向量, 但20是标量, 不能直接计算. 这个时候可以使用np.vectorize将函数向量化

  ```python
  avg_2_mod_vec = np.vectorize(avg_2_mod)
  avg_2_mod_vec(df['a'],df['b'])
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >array([15., nan, 35.])
  >```

- 使用装饰器

  ```python
  @np.vectorize
  def vec_avg_2_mod(x,y):
      if(x==20):
          return (np.NaN)
      else:
          return (x+y)/2
  vec_avg_2_mod(df['a'],df['b'])
  ```

  ><font color='red'>显示结果:</font>
  >
  >```shell
  >array([15., nan, 35.])
  >```

## 6 lambda函数

- 当函数比较简单的时候, 没有必要创建一个def 一个函数, 可以使用lambda表达式创建匿名函数

  ```python
  df.apply(lambda x: x+1)
  ```

  ><font color='red'>显示结果:</font>
  >
  >|      |    a |    b |
  >| ---: | ---: | ---: |
  >|    0 |   11 |   21 |
  >|    1 |   21 |   31 |
  >|    2 |   31 |   41 |

## 小结

- Series和DataFrame均可以通过apply传入自定义函数
- 有些时候需要通过np的vectorize函数才能进行向量化计算
# Pandas 数据类型

## 学习目标

- 了解Numpy的特点
- 应用Pandas 数据类型转换
- 掌握Pandas 分类数据类型使用方法

## 1 Pandas数据类型简介

### 1.1 Numpy 介绍

- Numpy（Numerical Python）是一个开源的Python科学计算库，用于快速处理任意维度的数组。

  Numpy支持常见的数组和矩阵操作。对于同样的数值计算任务，使用Numpy比直接使用Python要简洁的多。

  Numpy使用ndarray对象来处理多维数组，该对象是一个快速而灵活的大数据容器。

- NumPy提供了一个**N维数组类型ndarray**，它描述了**相同类型**的“items”的集合

  ![](img\score.png)

  用ndarray进行存储：

```python
import numpy as np

# 创建ndarray
score = np.array([[80, 89, 86, 67, 79],
[78, 97, 89, 67, 81],
[90, 94, 78, 67, 74],
[91, 91, 90, 67, 69],
[76, 87, 75, 67, 86],
[70, 79, 84, 67, 84],
[94, 92, 93, 67, 64],
[86, 85, 83, 67, 80]])

score
```

><font color='red'>显示结果</font>
>
>```shell
>array([[80, 89, 86, 67, 79],
>       [78, 97, 89, 67, 81],
>       [90, 94, 78, 67, 74],
>       [91, 91, 90, 67, 69],
>       [76, 87, 75, 67, 86],
>       [70, 79, 84, 67, 84],
>       [94, 92, 93, 67, 64],
>       [86, 85, 83, 67, 80]])
>```

- 使用Python列表可以存储一维数组，通过列表的嵌套可以实现多维数组，那么为什么还需要使用Numpy的ndarray呢？

**ndarray与Python原生list运算效率对比**

```python
import random
import time
import numpy as np
a = []
for i in range(100000000):
    a.append(random.random())
t1 = time.time()
sum1=sum(a)
t2=time.time()

b=np.array(a)
t4=time.time()
sum3=np.sum(b)
t5=time.time()
print(t2-t1, t5-t4)
```

t2-t1为使用python自带的求和函数消耗的时间，t5-t4为使用numpy求和消耗的时间

><font color='red'>显示结果</font>
>
>```shell
>0.6686017513275146 0.1469123363494873
>```

从结果中看到ndarray的计算速度要快很多，节约了时间

- Numpy专门针对ndarray的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，Numpy的优势就越明显。

**Numpy ndarray的优势**

1 数据在内存中存储的风格

![](img\numpy内存地址.png)

ndarray在存储数据时所有元素的类型都是相同的，数据内存地址是连续的，批量操作数组元素时速度更快

python原生list只能通过寻址方式找到下一个元素，这虽然也导致了在通用性方面Numpy的ndarray不及Python原生list，但计算的时候速度就慢了

2 ndarray支持并行化运算

3 Numpy底层使用C语言编写，内部解除了GIL（全局解释器锁），其对数组的操作速度不受Python解释器的限制，效率远高于纯Python代码

### 1.2 Numpy 的ndarray

- ndarray的属性

| 属性名字         | 属性解释                   |
| ---------------- | -------------------------- |
| ndarray.shape    | 数组维度的元组             |
| ndarray.ndim     | 数组维数                   |
| ndarray.size     | 数组中的元素数量           |
| ndarray.itemsize | 一个数组元素的长度（字节） |
| ndarray.dtype    | 数组元素的类型             |

- ndarray的形状

```python
a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2,3,4])
c = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
#打印形状
a.shape
b.shape
c.shape
```

><font color='red'>显示结果</font>
>
>```python
>(2, 3)
>(4,)  
>(2, 2, 3)
>```

- ndarray的类型

| 名称          | 描述                                              | 简写  |
| ------------- | ------------------------------------------------- | ----- |
| np.bool       | 用一个字节存储的布尔类型（True或False）           | 'b'   |
| np.int8       | 一个字节大小，-128 至 127                         | 'i'   |
| np.int16      | 整数，-32768 至 32767                             | 'i2'  |
| np.int32      | 整数，-2 **31 至 2** 32 -1                        | 'i4'  |
| np.int64      | 整数，-2 **63 至 2** 63 - 1                       | 'i8'  |
| np.uint8      | 无符号整数，0 至 255                              | 'u'   |
| np.uint16     | 无符号整数，0 至 65535                            | 'u2'  |
| np.uint32     | 无符号整数，0 至 2 ** 32 - 1                      | 'u4'  |
| np.uint64     | 无符号整数，0 至 2 ** 64 - 1                      | 'u8'  |
| np.float16    | 半精度浮点数：16位，正负号1位，指数5位，精度10位  | 'f2'  |
| np.float32    | 单精度浮点数：32位，正负号1位，指数8位，精度23位  | 'f4'  |
| np.float64    | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8'  |
| np.complex64  | 复数，分别用两个32位浮点数表示实部和虚部          | 'c8'  |
| np.complex128 | 复数，分别用两个64位浮点数表示实部和虚部          | 'c16' |
| np.object_    | python对象                                        | 'O'   |
| np.string_    | 字符串                                            | 'S'   |
| np.unicode_   | unicode类型                                       |       |

- 创建数组的时候指定类型

```python
a = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
a.dtype
```

><font color='red'>显示结果</font>
>
>```shell
>dtype('float32')
>```

### 1.3 Pandas的数据类型

- Pandas 是基于Numpy的，很多功能都依赖于Numpy的ndarray实现的，Pandas的数据类型很多与Numpy类似，属性也有很多类似

![](img\dtype.jpg)

- 查看数据类型

```python
import pandas as pd
tips = pd.read_csv('data/tips.csv')
tips.dtypes
```

><font color='red'>显示结果</font>
>
>```shell
>total_bill     float64
>tip            float64
>sex           category
>smoker        category
>day           category
>time          category
>size             int64
>dtype: object
>```

- 上面结果中，category数据类型表示分类变量，它与存储任意字符串的普通object数据类型不同。差异后面再讨论

## 2 类型转换

### 2.1 转换为字符串对象

- 在tips数据中，sex、smoker、day 和 time 变量都是category类型。通常，如果变量不是数值类型，应先将其转换成字符串类型以便后续处理
- 有些数据集中可能含有id列，id的值虽然是数字，但对id进行计算（求和，求平均等）没有任何意义，在某些情况下，可能需要把它们转换为字符串对象类型。
- 把一列的数据类型转换为字符串，可以使用astype方法。

```python
tips['sex_str'] = tips['sex'].astype(str)
```

- Python内置了str、float、int、complex和bool几种数据类型。此外还可以指定Numpy库支持的任何dtype，查看dtypes，会看到tips多出了object类型

```python
tips.dtypes
```

><font color='red'>显示结果</font>
>
>```shell
>total_bill     float64
>tip            float64
>sex           category
>smoker        category
>day           category
>time          category
>size             int64
>sex_str         object
>dtype: object
>```

### 2.2 转换为数值类型

- astype方法是通用函数，可用于把DataFrame中的任何列转换为其他dtype
- 可以向astype方法提供任何内置类型或numpy类型来转换列的数据类型

```python
#把total_bill转换成字符串
tips['total_bill'] = tips['total_bill'].astype(str)
tips.dtypes
```

><font color='red'>显示结果</font>
>
>```shell
>total_bill      object
>tip            float64
>sex           category
>smoker        category
>day           category
>time          category
>size             int64
>sex_str         object
>dtype: object
>```

```python
#把total_bill转换回float类型
tips['total_bill'] = tips['total_bill'].astype(float)
tips.dtypes
```

><font color='red'>显示结果</font>
>
>```shell
>total_bill     float64
>tip            float64
>sex           category
>smoker        category
>day           category
>time          category
>size             int64
>sex_str         object
>dtype: object
>```

#### to_numeric函数

- 如果想把变量转换为数值类型（int，float），还可以使用pandas的to_numeric函数
  - DataFrame每一列的数据类型必须相同，当有些数据中有缺失，但不是NaN时（如missing,null等），会使整列数据变成字符串类型而不是数值型，这个时候可以使用to_numeric处理

```python
#创造包含'missing'为缺失值的数据
tips_sub_miss = tips.head(10)
tips_sub_miss.loc[[1,3,5,7],'total_bill'] = 'missing'
tips_sub_miss
```

><font color='red'> 显示结果</font>
>
>|      | total_bill |  tip |    sex | smoker |  day |   time | size |
>| ---: | ---------: | ---: | -----: | -----: | ---: | -----: | ---: |
>|    0 |      16.99 | 1.01 | Female |     No |  Sun | Dinner |    2 |
>|    1 |    missing | 1.66 |   Male |     No |  Sun | Dinner |    3 |
>|    2 |      21.01 | 3.50 |   Male |     No |  Sun | Dinner |    3 |
>|    3 |    missing | 3.31 |   Male |     No |  Sun | Dinner |    2 |
>|    4 |      24.59 | 3.61 | Female |     No |  Sun | Dinner |    4 |
>|    5 |    missing | 4.71 |   Male |     No |  Sun | Dinner |    4 |
>|    6 |       8.77 | 2.00 |   Male |     No |  Sun | Dinner |    2 |
>|    7 |    missing | 3.12 |   Male |     No |  Sun | Dinner |    4 |
>|    8 |      15.04 | 1.96 |   Male |     No |  Sun | Dinner |    2 |
>|    9 |      14.78 | 3.23 |   Male |     No |  Sun | Dinner |    2 |

```python
#查看数据类型 dtypes 会发现total_bill列变成了字符串对象类型
tips_sub_miss.dtypes
```

><font color='red'> 显示结果</font>
>
>```shell
>total_bill      object
>tip            float64
>sex           category
>smoker        category
>day           category
>time          category
>size             int64
>dtype: object
>```

- 对上面的数据集使用astype方法把total_bill 列转换回float类型,会抛错, Pandas 无法把'missing'转换成float

```python
tips_sub_miss['total_bill'].astype(float)
```

><font color='red'> 显示结果</font>
>
>```shell
>---------------------------------------------------------------------------
>ValueError                                Traceback (most recent call last)
><ipython-input-8-3aba35b22fb4> in <module>
>----> 1 tips_sub_miss['total_bill'].astype(float)
>... ....
>
>ValueError: could not convert string to float: 'missing'
>```

- 如果使用Pandas库中的to_numeric函数进行转换,也会得到类似的错误

```python
pd.to_numeric(tips_sub_miss['total_bill'])
```

><font color='red'> 显示结果</font>
>
>```shell
>ValueError                                Traceback (most recent call last)
>pandas\_libs\lib.pyx in pandas._libs.lib.maybe_convert_numeric()
>
>ValueError: Unable to parse string "missing"
>
>During handling of the above exception, another exception occurred:
>
>ValueError                                Traceback (most recent call last)
><ipython-input-9-4fcf9a4ed513> in <module>
>----> 1 pd.to_numeric(tips_sub_miss['total_bill'])
>
>~\anaconda3\lib\site-packages\pandas\core\tools\numeric.py in to_numeric(arg, errors, downcast)
>    148         try:
>    149             values = lib.maybe_convert_numeric(
>--> 150                 values, set(), coerce_numeric=coerce_numeric
>    151             )
>    152         except (ValueError, TypeError):
>
>pandas\_libs\lib.pyx in pandas._libs.lib.maybe_convert_numeric()
>
>ValueError: Unable to parse string "missing" at position 1
>```

- to_numeric函数有一个参数errors,它决定了当该函数遇到无法转换的数值时该如何处理
  - 默认情况下,该值为raise,如果to_numeric遇到无法转换的值时,会抛错
  - coerce: 如果to_numeric遇到无法转换的值时,会返回NaN
  - ignore: 如果to_numeric遇到无法转换的值时会放弃转换,什么都不做

```python
pd.to_numeric(tips_sub_miss['total_bill'],errors = 'ignore')
```

><font color='red'> 显示结果</font>
>
>```shell
>0      16.99
>1    missing
>2      21.01
>3    missing
>4      24.59
>5    missing
>6       8.77
>7    missing
>8      15.04
>9      14.78
>Name: total_bill, dtype: object
>```

```python
pd.to_numeric(tips_sub_miss['total_bill'],errors = 'coerce')
```

><font color='red'> 显示结果</font>
>
>```shell
>0    16.99
>1      NaN
>2    21.01
>3      NaN
>4    24.59
>5      NaN
>6     8.77
>7      NaN
>8    15.04
>9    14.78
>Name: total_bill, dtype: float64
>```

#### to_numeric向下转型

- to_numeric函数还有一个downcast参数, downcast接受的参数为 'integer','signed','float','unsigned'
- downcast参数设置为float之后, total_bill的数据类型由float64变为float32

```python
pd.to_numeric(tips_sub_miss['total_bill'],errors = 'coerce',downcast='float')
```

><font color='red'> 显示结果</font>
>
>```shell
>0    16.99
>1      NaN
>2    21.01
>3      NaN
>4    24.59
>5      NaN
>6     8.77
>7      NaN
>8    15.04
>9    14.78
>Name: total_bill, dtype: float32
>```

- 从上面的结果看出,转换之后的数据类型为float32, 意味着占用的内存更小了

## 3 分类数据(category)

- Pandas 有一种类别数据, category,用于对分类值进行编码

### 3.1 转换为category类型

```python
tips['sex'] = tips['sex'].astype('str')
tips.info()
```

><font color='red'> 显示结果</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 244 entries, 0 to 243
>Data columns (total 7 columns):
> #   Column      Non-Null Count  Dtype  
>---  ------      --------------  -----  
> 0   total_bill  244 non-null    float64
> 1   tip         244 non-null    float64
> 2   sex         244 non-null    object 
> 3   smoker      244 non-null    object 
> 4   day         244 non-null    object 
> 5   time        244 non-null    object 
> 6   size        244 non-null    int64  
>dtypes: float64(2), int64(1), object(4)
>memory usage: 13.5+ KB
>```

```python
tips['sex'] = tips['sex'].astype('category')
tips.info()
```

><font color='red'> 显示结果</font>
>
>```shell
><class 'pandas.core.frame.DataFrame'>
>RangeIndex: 244 entries, 0 to 243
>Data columns (total 7 columns):
> #   Column      Non-Null Count  Dtype   
>---  ------      --------------  -----   
> 0   total_bill  244 non-null    float64 
> 1   tip         244 non-null    float64 
> 2   sex         244 non-null    category
> 3   smoker      244 non-null    object  
> 4   day         244 non-null    object  
> 5   time        244 non-null    object  
> 6   size        244 non-null    int64   
>dtypes: category(1), float64(2), int64(1), object(3)
>memory usage: 11.9+ KB
>```

### 3.2 利用category排序

- 利用pd.Categorical()创建categorical数据，Categorical()常用三个参数

  - 参数一 values，如果values中的值，不在categories参数中，会被NaN代替
  - 参数二 categories，指定可能存在的类别数据
  - 参数三 ordered, 是否指定顺序

  ```python
  s = pd.Series(pd.Categorical(["a", "b", "c", "d"],categories=["c", "b", "a"]))
  ```

  ><font color='red'> 显示结果</font>
  >
  >```shell
  >0      a
  >1      b
  >2      c
  >3    NaN
  >dtype: category
  >Categories (3, object): ['c', 'b', 'a']
  >```

- category排序

  - 准备数据

  ```python
  #创建categorical型Series
  series_cat = pd.Series(['B','D','C','A'], dtype='category')
  series_cat
  ```

  ><font color='red'> 显示结果</font>
  >
  >```shell
  >0    B
  >1    D
  >2    C
  >3    A
  >dtype: category
  >Categories (4, object): ['A', 'B', 'C', 'D']
  >```

  - 对数据排序

  ```python
  series_cat.sort_values()
  ```

  ><font color='red'> 显示结果</font>
  >
  >```shell
  >3    A
  >0    B
  >2    C
  >1    D
  >dtype: category
  >Categories (4, object): ['A', 'B', 'C', 'D']
  >```

  - 指定顺序 CategoricalDtype

  ```python
  from pandas.api.types import CategoricalDtype
  cat = CategoricalDtype(categories=['B','D','A','C'],ordered=True)
  series_cat1 = series_cat.astype(cat)
  series_cat.sort_values()
  ```

  ><font color='red'> 显示结果</font>
  >
  >```shell
  >3    A
  >0    B
  >2    C
  >1    D
  >dtype: category
  >Categories (4, object): ['A', 'B', 'C', 'D']
  >```

  ```python
  series_cat1.sort_values()
  ```

  ><font color='red'> 显示结果</font>
  >
  >```shell
  >0    B
  >1    D
  >3    A
  >2    C
  >dtype: category
  >Categories (4, object): ['B' < 'D' < 'A' < 'C']
  >```

  - 想要临时修改排序规则，可以使用.cat.reorder_categories()方法：

  ```python
  series_cat.cat.reorder_categories(['D','B','C','A'],ordered=True,
                                    inplace=True)#inplace参数设置为True使得变动覆盖原数据
  series_cat.sort_values()
  ```

  ><font color='red'> 显示结果</font>
  >
  >```shell
  >1    D
  >0    B
  >2    C
  >3    A
  >dtype: category
  >Categories (4, object): ['D' < 'B' < 'C' < 'A']
  >```

## 小结

- Numpy的特点
  - Numpy是一个高效科学计算库，Pandas的数据计算功能是对Numpy的封装
  - ndarray是Numpy的基本数据结构，Pandas的Series和DataFrame好多函数和属性都与ndarray一样
  - Numpy的计算效率比原生Python效率高很多，并且支持并行计算
- Pandas 数据类型转换
  - Pandas除了数值型的int 和 float类型外，还有object ，category，bool，datetime类型
  - 可以通过as_type 和 to_numeric 函数进行数据类型转换
- Pandas 分类数据类型
  - category类型，可以用来进行排序，并且可以自定义排序顺序
  - CategoricalDtype可以用来定义顺序
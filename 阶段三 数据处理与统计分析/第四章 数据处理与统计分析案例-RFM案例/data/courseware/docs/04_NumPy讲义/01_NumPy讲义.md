# 1 NumPy

## 学习目标

- 了解NumPy特性
- 掌握 NumPy的使用方法

## 1 NumPy简介

NumPy（Numerical Python）是Python数据分析必不可少的第三方库，NumPy的出现一定程度上解决了Python运算性能不佳的问题，同时提供了更加精确的数据类型，使其具备了构造复杂数据类型的能力。本身是由C语言开发，是个很基础的扩展，NumPy被Python其它科学计算包作为基础包，因此理解np的数据类型对python数据分析十分重要。NumPy重在数值计算，主要用于多维数组（矩阵）处理的库。用来存储和处理大型矩阵，比Python自身的嵌套列表结构要高效的多。

NumPy重要功能如下：

1. 高性能科学计算和数据分析的基础包

2. ndarray，多维数组，具有矢量运算能力，快速、节省空间

3. 矩阵运算，无需循环，可完成类似Matlab中的矢量运算

4. 用于读写磁盘数据的工具以及用于操作内存映射文件的工具

官网地址[NumPy](https://numpy.org/)

## 2 NumPy属性

NumPy的主要对象是同类型元素的多维数组，所有的元素都是一种类型。在NumPy中维度(dimensions)叫做轴(axes)，轴的个数叫做秩(rank)。

例如，在3D空间一个点的坐标[1, 2, 3]是一个秩为1的数组，因为它只有一个轴。那个轴长度为3。又例如，在以下例子中，数组的秩为2(它有两个维度)，第一个维度长度为3，第二个维度长度为3。

[[1., 0., 0.],

[0., 1., 2.]]

NumPy的数组类被称作ndarray，通常被称作数组。注意NumPy.array和标准Python库类array.array并不相同，后者只处理一维数组和提供少量功能，而前者可以进行高维数组的创建和提供较多且复杂的运算功能。更多重要ndarray对象属性有：

- ndarray.ndim

数组轴的个数

- ndarray.shape

数组的维度。这是一个指示数组在每个维度上大小的整数元组。例如一个n排m列的矩阵，它的shape属性将是(2,3),这个元组的长度显然是秩，即维度或者ndim属性。

- ndarray.size

数组元素的总个数，等于shape属性中元组元素的乘积。

- ndarray.dtype

一个用来描述数组中元素类型的对象，可以通过创造或指定dtype使用标准Python类型。另外NumPy提供它自己的数据类型。

- ndarray.itemsize

数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsize属性值为8(=64/8),又如，一个元素类型为complex32的数组itemsize属性为4(=32/8).

- ndarray.data

包含实际数组元素的缓冲区，通常我们不需要使用这个属性，因为我们总是通过索引来使用数组中的元素。

代码示例：

```python
import numpy as np
a = np.arange(15).reshape(3, 5)
print ("数组的维度:",a.shape)
print ("数组轴的个数:",a.ndim)
print ("数组元素类型:",a.dtype)
print ("数组中每个元素的字节大小:",a.itemsize)
print ("数组元素的总个数:", a.size)
print ("类型查询:",type(a))

#创建一个数组
b = np.array([6, 7, 8])
print ("数组b:",b)
print ("数组b类型:",type(b))

运行结果:
数组的维度: (3, 5)
数组轴的个数: 2
数组元素类型: int32
数组中每个元素的字节大小: 4
数组元素的总个数: 15
类型查询: <class 'numpy.ndarray'>
数组b: [6 7 8]
数组b类型: <class 'numpy.ndarray'>
```

## 3 创建ndarray

ndarray 多维数组(N Dimension Array)

NumPy数组是一个多维的数组对象（矩阵），称为ndarray，具有矢量算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点。注意：ndarray的下标从0开始，且数组里的所有元素必须是相同类型。

### 1 array() 

```python
import numpy as np
a = np.array([2, 3, 4])
print ("数组a元素类型:",a)
print ("数组a类型:",a.dtype)

b = np.array([1.2, 3.5, 5.1])
print ("数组b元素类型:",b.dtype)
```

### 2 zeros() /ones()/empty()

函数function创建一个全是0的数组，函数ones创建一个全1的数组，函数empty创建一个内容随机并且依赖于内存状态的数组。默认创建的数组类型(dtype)都是float64

```python
zeros1=np.zeros( (3,4) )
print ("数组zeros1:",zeros1)
ones1=np.ones((2,3,4))
print ("数组ones1:",ones1)
empty1 = np.empty((2, 3))
print ("数组empty1:",empty1)
```

### 3 arange()

arange() 类似 python 的 range() ，创建一个一维 ndarray 数组。

```python
np_arange = np.arange(10, 20, 5,dtype=int)
print ("arange创建np_arange:",np_arange)
print( "arange创建np_arange的元素类型:",np_arange.dtype)
print ("arange创建np_arange的类型:",type(np_arange))
```

### 4 matrix()

matrix 是 ndarray 的子类，只能生成 2 维的矩阵

```python
x1=np.mat("1 2;3 4")
print( x1)

x2=np.matrix("1,2;3,4")
print( x2)
x3=np.matrix([[1,2,3,4],[5,6,7,8]])
print( x3)
```

### 5 创建随机数矩阵

```python
import numpy as np

# 生成指定维度大小（3行4列）的随机多维浮点型数据（二维），rand固定区间0.0 ~ 1.0
arr = np.random.rand(3, 4)
print(arr)
print(type(arr))

# 生成指定维度大小（3行4列）的随机多维整型数据（二维），randint()可以指定区间（-1, 5）
arr = np.random.randint(-1, 5, size = (3, 4)) 
print(arr)
print(type(arr))

# 生成指定维度大小（3行4列）的随机多维浮点型数据（二维），uniform()可以指定区间（-1, 5）产生-1到5之间均匀分布的样本值
arr = np.random.uniform(-1, 5, size = (3, 4)) # 
print(arr)
print(type(arr))
```


### 6 ndarray的数据类型

1. dtype参数，指定数组的数据类型，类型名+位数，如float64, int32

2. astype方法，转换数组的数据类型

```python
# 初始化3行4列数组，数据类型为float64
zeros_float_arr = np.zeros((3, 4), dtype=np.float64)
print(zeros_float_arr)
print(zeros_float_arr.dtype) #float64

# astype转换数据类型，将已有的数组的数据类型转换为int32
zeros_int_arr = zeros_float_arr.astype(np.int32)
print(zeros_int_arr)
print(zeros_int_arr.dtype)  #int32
```
### 7 等比/等差数列

#### np.logspace等比数列

logspace中，开始点和结束点是10的幂

我们让开始点为0，结束点为0，元素个数为10，看看输出结果。

```python
a = np.logspace(0,0,10)
a
# 输出结果
# array([ 1., 1., 1., 1., 1., 1.,  1., 1., 1., 1.])
```

● 我们看下面的例子，0代表10的0次方，9代表10的9次方。

```python
a = np.logspace(0,9,10)
a
# 输出结果
# array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,
#       1.e+08, 1.e+09])
```

● 假如，我们想要改变基数，不让它以10为底数，我们可以改变base参数，将其设置为2试试。

```python
a = np.logspace(0,9,10,base=2)
a
# 输出结果
# array([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
```

#### np.linspace等差数列

np.linspace是用于创建一个一维数组，并且是等差数列构成的一维数组，它最常用的有三个参数。

● 第一个例子，用到三个参数，第一个参数表示起始点，第二个参数表示终止点，第三个参数表示数列的个数。

```python
a = np.linspace(1,10,10)
a
# 输出结果
# array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
```

● linspace创建的数组元素是浮点型。

```python
a.dtype
# 输出结果
# dtype('float64')
```

● 可以使用参数endpoint来决定是否包含终止值，默认值为True。

```python
a = np.linspace(1,10,10,endpoint=False)
a
# 输出结果

# array([ 1. , 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1])
```



## 4 NumPy内置函数

### 1 基本函数

- np.ceil(): 向上最接近的整数，参数是 number 或 array

- np.floor(): 向下最接近的整数，参数是 number 或 array

- np.rint(): 四舍五入，参数是 number 或 array

- np.isnan(): 判断元素是否为 NaN(Not a Number)，参数是 number 或 array

- np.multiply(): 元素相乘，参数是 number 或 array

- np.divide(): 元素相除，参数是 number 或 array

- np.abs()：元素的绝对值，参数是 number 或 array

- np.where(condition, x, y): 三元运算符，x if condition else y

```python
# randn() 返回具有标准正态分布的序列。
arr = np.random.randn(2,3)

print(arr)

print(np.ceil(arr))

print(np.floor(arr))

print(np.rint(arr))

print(np.isnan(arr))

print(np.multiply(arr, arr))

print(np.divide(arr, arr))

print(np.where(arr > 0, 1, -1))
```



### 2 统计函数

- np.mean(), np.sum()：所有元素的平均值，所有元素的和，参数是 number 或 array

- np.max(), np.min()：所有元素的最大值，所有元素的最小值，参数是 number 或 array

- np.std(), np.var()：所有元素的标准差，所有元素的方差，参数是 number 或 array

- np.argmax(), np.argmin()：最大值的下标索引值，最小值的下标索引值，参数是 number 或 array

- np.cumsum(), np.cumprod()：返回一个一维数组，每个元素都是之前所有元素的 累加和 和 累乘积，参数是 number 或 array

  多维数组默认统计全部维度，axis参数可以按指定轴心统计，值为0则按列统计，值为1则按行统计。

```python
arr = np.arange(12).reshape(3,4)

print(arr)

print(np.cumsum(arr)) # 返回一个一维数组，每个元素都是之前所有元素的 累加和

print(np.sum(arr)) # 所有元素的和

print(np.sum(arr, axis=0)) # 数组的按列统计和

print(np.sum(arr, axis=1)) # 数组的按行统计和
```



### 3 比较函数

假如我们想要知道矩阵a和矩阵b中所有对应元素是否相等，我们需要使用all方法，假如我们想要知道矩阵a和矩阵b中对应元素是否有一个相等，我们需要使用any方法。

- np.any(): 至少有一个元素满足指定条件，返回True
- np.all(): 所有的元素满足指定条件，返回True

```python
arr = np.random.randn(2,3)

print(arr)

print(np.any(arr > 0))

print(np.all(arr > 0))
```



### 4 去重函数

np.unique():找到唯一值并返回排序结果，类似于Python的set集合

```python
arr = np.array([[1, 2, 1], [2, 3, 4]])
print(arr)
print(np.unique(arr))
```



### 5 排序函数

对数组元素进行排序

```python
arr = np.array([1, 2, 34, 5])
print ("原数组arr:",arr)

#np.sort()函数排序，返回排序后的副本
sortarr1= np.sort(arr)
print ("numpy.sort()函数排序后的数组:",sortarr1)

# ndarray直接调用sort，在原数据上进行修改
arr.sort()
print ("数组.sort()方法排序：",arr)

```



## 5 NumPy运算

### 1 基本运算

数组的算数运算是按照元素的。新的数组被创建并且被结果填充。

```python
import numpy as np
a = np.array([20,30,40,50])
b = np.arange(4)
c = a-b
print("数组a:",a)
print("数组b:",b)
print("数组运算a-b:",c)
```

### 2 矩阵乘法

```python
import numpy as np

x=np.array([[1,2,3],[4,5,6]])
y=np.array([[6,23],[-1,7],[8,9]])

print(x)
print(y)

print(x.dot(y))
print(np.dot(x,y))

a = np.array([[1,2,3], [4,5,6]])
b = np.array([[2,2,2],[3,3,3]])
print(a*b)
```

## 小结

- NumPy常见属性
- NumPy创建方法
- NumPy内置函数

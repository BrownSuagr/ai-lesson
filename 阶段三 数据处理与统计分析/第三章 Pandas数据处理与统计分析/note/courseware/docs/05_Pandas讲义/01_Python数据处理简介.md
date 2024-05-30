# 1 Python 数据分析处理简介

## 学习目标

- 了解Python做数据分析和处理的优势
- 知道Python数据分析和处理常用开源库

## 1 为什么使用Python进行数据分析和处理

### 1.1 使用Python进行数据分析和处理的优势

<img src="img/pic1.png" style="zoom: 80%;" />

- Python作为当下最为流行的编程语言之一，可以独立完成数据分析处理的各种任务
    - 功能强大，在数据分析处理领域里有海量开源库，并持续更新
    - 是当下热点——机器学习/深度学习 领域最热门的编程语言
    - 除数据分析和处理领域外，在爬虫，Web开发等领域均有应用

    

- 与Excel，PowerBI，Tableau等软件比较

    - Excel有百万行数据限制，PowerBI ，Tableau在处理大数据的时候速度相对较慢
    - Excel，Power BI 和Tableau 需要付费购买授权
    - Python作为热门编程语言，功能远比Excel，PowerBI，Tableau等软件强大
    - Python跨平台，Windows，MacOS，Linux都可以运行

    

- 与R语言比较

    - Python在处理海量数据的时候比R语言效率更高
    - Python的工程化能力更强，应用领域更广泛，R专注于统计与数据分析处理领域
    - Python在非结构化数据（文本，音视频，图像）和深度学习领域比R更具有优势
    - 在数据分析处理相关开源社区，python相关的内容远多于R语言

  

## 2 常用Python数据分析和处理开源库介绍

### 2.1 Numpy<img src="img/numpy-logo-300.png" alt="img" style="zoom:50%;" />

- NumPy(Numerical Python) 是 Python 语言的一个扩展程序库
- 是一个运行速度非常快的数学库，主要用于数组计算，包含：
    - 一个强大的N维数组对象 ndarray
    - 广播功能函数
    - 整合 C/C++/Fortran 代码的工具
    - 线性代数、傅里叶变换、随机数生成等功能

### 2.2 Pandas   ![image-20200427174034262](img/pic2.png)

- Pandas是一个强大的分析结构化数据的工具集
    - 它的使用基础是Numpy（提供高性能的矩阵运算）
    - 用于数据挖掘和数据分析和处理，同时也提供数据清洗功能
- Pandas利器之 Series
    - 它是一种类似于一维数组的对象
    - 是由一组数据(各种NumPy数据类型)以及一组与之相关的数据标签(即索引)组成
    - 仅由一组数据也可产生简单的Series对象
- Pandas利器之 DataFrame
    - DataFrame是Pandas中的一个表格型的数据结构
    - 包含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔型等)
    - DataFrame即有行索引也有列索引，可以被看做是由Series组成的字典

### 2.3 Matplotlib <img src="img/logo2_compressed.svg" alt="matplotlib" style="zoom: 20%;" />

- Matplotlib 是一个功能强大的数据可视化开源Python库
- Python中使用最多的图形绘图库
- 可以创建静态, 动态和交互式的图表

### 2.4 Seaborn 

- Seaborn是一个Python数据可视化开源库
- 建立在matplotlib之上，并集成了pandas的数据结构
- Seaborn通过更简洁的API来绘制信息更丰富，更具吸引力的图像
-  面向数据集的API，与Pandas配合使用起来比直接使用Matplotlib更方便

### 2.5 Sklearn ![logo](img/scikit-learn-logo-small.png)

- scikit-learn 是基于 Python 语言的机器学习工具
    - 简单高效的数据挖掘和数据分析和处理工具
    - 可供大家在各种环境中重复使用
    - 建立在 NumPy ，SciPy 和 matplotlib 上

### 2.6 Jupyter Notebook/Lab  ![jupyter logo](img/main-logo.svg)

- Jupyter Notebook是一个开源Web应用程序，使用Jupyter Notebook可以创建和共享
    - 代码
    - 数学公式
    - 可视化图表
    - 笔记文档
- Jupyter Notebook用途
    - 数据清理和转换
    - 数值模拟
    - 统计分析
    - 数据可视化
    - 机器学习等
- Jupyter Notebook是数据分析处理学习和开发的首选开发环境
- JupyterLab是Jupyter Notebook升级版



## 3 数据分析和处理流程

![数据分析和处理流程](img/数据分析流程.svg)

## 小结

- 了解Python做数据分析和处理的优势
  
  - Python可以独立高效的完成数据分析和处理相关的全部工作
- 知道Python数据分析和处理常用开源库
  
  - Pandas，Numpy，Matplotlib，Seaborn，SKlearn，Jupyter Notebook

### 学习目标

* 了解fasttext工具的作用.
* 了解fasttext工具的优势及其原因.
* 掌握fasttext的安装方法.



<center><img src="./img/fasttext-logo-color-web.png" height="auto" width="auto"/></center>



## 1 fasttext介绍

### 1.1 fasttext作用

作为NLP工程领域常用的工具包, fasttext有两大作用:

* 进行文本分类
* 训练词向量



### 1.2 fasttext工具包的优势

* 正如它的名字, 在保持较高精度的情况下, 快速的进行训练和预测是fasttext的最大优势.


* fasttext优势的原因:
  * fasttext工具包中内含的fasttext模型具有十分简单的网络结构.
  * 使用fasttext模型训练词向量时使用层次softmax结构, 来提升超多类别下的模型性能.
  * 由于fasttext模型过于简单无法捕捉词序特征, 因此会进行n-gram特征提取以弥补模型缺陷提升精度.



### 1.3 fasttext的安装

```shell
pip install fasttext
```



### 1.4 验证安装

```python
Python 3.8.12 (default) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import fasttext
>>>
```


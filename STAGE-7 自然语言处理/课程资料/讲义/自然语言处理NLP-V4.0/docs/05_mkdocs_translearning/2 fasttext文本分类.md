### 学习目标

* 了解什么是文本分类及其种类.
* 了解fasttext分类模型的架构
* 掌握fasttext工具进行文本分类的过程.

## 1 文本分类介绍

### 1.1 文本分类概念

* 文本分类的是将文档（例如电子邮件，帖子，文本消息，产品评论等）分配给一个或多个类别. 当今文本分类的实现多是使用机器学习方法从训练数据中提取分类规则以进行分类, 因此构建文本分类器需要带标签的数据.

### 1.2 文本分类种类

* 二分类: 
    * 文本被分类两个类别中, 往往这两个类别是对立面, 比如: 判断一句评论是好评还是差评.
* 单标签多分类: 
    * 文本被分入到多个类别中, 且每条文本只能属于某一个类别(即被打上某一个标签), 比如: 输入一个人名, 判断它是来自哪个国家的人名. 
* 多标签多分类: 
    * 文本被分人到多个类别中, 但每条文本可以属于多个类别(即被打上多个标签), 比如: 输入一段描述, 判断可能是和哪些兴趣爱好有关, 一段描述中可能即讨论了美食, 又太讨论了游戏爱好.

## 2 Fasttext模型架构

- FastText 模型架构和 Word2Vec 中的 CBOW 模型很类似, 不同之处在于, FastText 预测标签, 而 CBOW 模型预测中间词.


- FastText的模型分为三层架构: 
  - 输入层: 是对文档embedding之后的向量, 包含N-gram特征
  - 隐藏层: 是对输入数据的求和平均
  - 输出层: 是文档对应的label

<div align=center><img src="./img/image-20221224222050666.png" style="zoom:45%" ><img/></div>

------

## 3 文本分类的过程

* 第一步: 获取数据
* 第二步: 训练集与验证集的划分
* 第三步: 训练模型
* 第四步: 使用模型进行预测并评估
* 第五步: 模型调优
* 第六步: 模型保存与重加载 

------

### 3.1 获取数据

数据集介绍，本案例烹饪相关的数据集, 它是由facebook AI实验室提供的演示数据集

```shell
# 数据集在虚拟机/root/data/cooking下

# 查看数据的前10条
$ head cooking.stackexchange.txt
```

```text
__label__sauce __label__cheese How much does potato starch affect a cheese sauce recipe?
__label__food-safety __label__acidity Dangerous pathogens capable of growing in acidic environments
__label__cast-iron __label__stove How do I cover up the white spots on my cast iron stove?
__label__restaurant Michelin Three Star Restaurant; but if the chef is not there
__label__knife-skills __label__dicing Without knife skills, how can I quickly and accurately dice vegetables?
__label__storage-method __label__equipment __label__bread What's the purpose of a bread box?
__label__baking __label__food-safety __label__substitutions __label__peanuts how to seperate peanut oil from roasted peanuts at home?
__label__chocolate American equivalent for British chocolate terms
__label__baking __label__oven __label__convection Fan bake vs bake
__label__sauce __label__storage-lifetime __label__acidity __label__mayonnaise Regulation and balancing of readymade packed mayonnaise and other sauces
```

> * 数据说明:
> * cooking.stackexchange.txt中的每一行都包含一个标签列表，后跟相应的文档, 标签列表以类似"__label__sauce __label__cheese"的形式展现, 代表有两个标签sauce和cheese, 所有标签__label__均以前缀开头，这是fastText识别标签或单词的方式. 标签之后的一段话就是文本信息.如: How much does potato starch affect a cheese sauce recipe?



### 3.2 训练集与验证集的划分

```python
# 查看数据总数
$ wc cooking.stackexchange.txt 

15404  169582 1401900 cooking.stackexchange.txt 
# 多少行 多少单词 占用字节数(多大)
```

```shell
# 12404条数据作为训练数据
$ head -n 12404 cooking.stackexchange.txt > cooking.train
# 3000条数据作为验证数据
$ tail -n 3000 cooking.stackexchange.txt > cooking.valid
```



### 3.3 训练模型

```python
# 导入fasttext
import fasttext
# 使用fasttext的train_supervised方法进行文本分类模型的训练
model = fasttext.train_supervised(input="data/cooking/cooking.train")
```

> - 结果输出

```python
# 获得结果
Read 0M words
# 不重复的词汇总数
Number of words:  14543
# 标签总数
Number of labels: 735
# Progress: 训练进度, 因为我们这里显示的是最后的训练完成信息, 所以进度是100%
# words/sec/thread: 每个线程每秒处理的平均词汇数
# lr: 当前的学习率, 因为训练完成所以学习率是0
# avg.loss: 训练过程的平均损失 
# ETA: 预计剩余训练时间, 因为已训练完成所以是0
Progress: 100.0% words/sec/thread:   60162 lr:  0.000000 avg.loss: 10.056812 ETA:   0h 0m 0s
```



### 3.4 使用模型进行预测并评估

```python
# 使用模型预测一段输入文本, 通过我们常识, 可知预测是正确的, 但是对应预测概率并不大
>>> model.predict("Which baking dish is best to bake a banana bread ?")
# 元组中的第一项代表标签, 第二项代表对应的概率
(('__label__baking',), array([0.06550845]))
```

```
# 通过我们常识可知预测是错误的
>>> model.predict("Why not put knives in the dishwasher?")
(('__label__food-safety',), array([0.07541209]))
```

```python
# 为了评估模型到底表现如何, 我们在3000条的验证集上进行测试
>>> model.test("data/cooking/cooking.valid")
# 元组中的每项分别代表, 验证集样本数量, 精度以及召回率 
# 我们看到模型精度和召回率表现都很差, 接下来我们讲学习如何进行优化.
(3000, 0.124, 0.0541)
```



### 3.5 模型调优

#### 1 原始数据处理:

```text
# 通过查看数据, 我们发现数据中存在许多标点符号与单词相连以及大小写不统一, 
# 这些因素对我们最终的分类目标没有益处, 反是增加了模型提取分类规律的难度,
# 因此我们选择将它们去除或转化

# 处理前的部分数据
__label__fish Arctic char available in North-America
__label__pasta __label__salt __label__boiling When cooking pasta in salted water how much of the salt is absorbed?
__label__coffee Emergency Coffee via Chocolate Covered Coffee Beans?
__label__cake Non-beet alternatives to standard red food dye
__label__cheese __label__lentils Could cheese "halt" the tenderness of cooking lentils?
__label__asian-cuisine __label__chili-peppers __label__kimchi __label__korean-cuisine What kind of peppers are used in Gochugaru ()?
__label__consistency Pavlova Roll failure
__label__eggs __label__bread What qualities should I be looking for when making the best French Toast?
__label__meat __label__flour __label__stews __label__braising Coating meat in flour before browning, bad idea?
__label__food-safety Raw roast beef on the edge of safe?
__label__pork __label__food-identification How do I determine the cut of a pork steak prior to purchasing it?
```

```shell
# 通过服务器终端进行简单的数据预处理
# 使标点符号与单词分离并统一使用小写字母
>> cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
>> head -n 12404 cooking.preprocessed.txt > cooking.pre.train
>> tail -n 3000 cooking.preprocessed.txt > cooking.pre.valid
```

```text
# 处理后的部分数据
__label__fish arctic char available in north-america
__label__pasta __label__salt __label__boiling when cooking pasta in salted water how much of the salt is absorbed ?
__label__coffee emergency coffee via chocolate covered coffee beans ?
__label__cake non-beet alternatives to standard red food dye
__label__cheese __label__lentils could cheese "halt" the tenderness of cooking lentils ?
__label__asian-cuisine __label__chili-peppers __label__kimchi __label__korean-cuisine what kind of peppers are used in gochugaru  (  )  ?
__label__consistency pavlova roll failure
__label__eggs __label__bread what qualities should i be looking for when making the best french toast ?
__label__meat __label__flour __label__stews __label__braising coating meat in flour before browning ,  bad idea ?
__label__food-safety raw roast beef on the edge of safe ?
__label__pork __label__food-identification how do i determine the cut of a pork steak prior to purchasing it ?
```

#### 2 数据处理后进行训练并测试:

```python
# 重新训练
>>> model = fasttext.train_supervised(input="data/cooking/cooking.pre.train")
Read 0M words

# 不重复的词汇总数减少很多, 因为之前会把带大写字母或者与标点符号相连接的单词都认为是新的单词
Number of words:  8952
Number of labels: 735

# 我们看到平均损失有所下降
Progress: 100.0% words/sec/thread:   65737 lr:  0.000000 avg.loss:  9.966091 ETA:   0h 0m 0s

# 重新测试
>>> model.test("data/cooking/cooking.pre.valid")
# 我们看到精度和召回率都有所提升
(3000, 0.161, 0.06962663975782038)
```

#### 3 增加训练轮数:

```python
# 设置train_supervised方法中的参数epoch来增加训练轮数, 默认的轮数是5次
# 增加轮数意味着模型能够有更多机会在有限数据中调整分类规律, 当然这也会增加训练时间
>>> model = fasttext.train_supervised(input="data/cooking/cooking.pre.train", epoch=25)
Read 0M words
Number of words:  8952
Number of labels: 735

# 我们看到平均损失继续下降
Progress: 100.0% words/sec/thread:   66283 lr:  0.000000 avg.loss:  7.203885 ETA:   0h 0m 0s

>>> model.test("data/cooking/cooking.pre.valid")
# 我们看到精度已经提升到了42%, 召回率提升至18%.
(3000, 0.4206666666666667, 0.1819230214790255)
```



#### 4 调整学习率:

```python
# 设置train_supervised方法中的参数lr来调整学习率, 默认的学习率大小是0.1
# 增大学习率意味着增大了梯度下降的步长使其在有限的迭代步骤下更接近最优点
>>> model = fasttext.train_supervised(input="data/cooking/cooking.pre.train", lr=1.0, epoch=25)
Read 0M words
Number of words:  8952
Number of labels: 735

# 平均损失继续下降
Progress: 100.0% words/sec/thread:   66027 lr:  0.000000 avg.loss:  4.278283 ETA:   0h 0m 0s

>>> model.test("data/cooking/cooking.pre.valid")
# 我们看到精度已经提升到了47%, 召回率提升至20%.
(3000, 0.47633333333333333, 0.20599682860025947)
```



#### 5 增加n-gram特征:

```python
# 设置train_supervised方法中的参数wordNgrams来添加n-gram特征, 默认是1, 也就是没有n-gram特征
# 我们这里将其设置为2意味着添加2-gram特征, 这些特征帮助模型捕捉前后词汇之间的关联, 更好的提取分类规则用于模型分类, 当然这也会增加模型训时练占用的资源和时间.
>>> model = fasttext.train_supervised(input="data/cooking/cooking.pre.train", lr=1.0, epoch=25, wordNgrams=2)
Read 0M words
Number of words:  8952
Number of labels: 735

# 平均损失继续下降
Progress: 100.0% words/sec/thread:   65084 lr:  0.000000 avg.loss:  3.189422 ETA:   0h 0m 0s

>>> model.test("data/cooking/cooking.pre.valid")
# 我们看到精度已经提升到了49%, 召回率提升至21%.
(3000, 0.49233333333333335, 0.2129162462159435)
```



#### 6 修改损失计算方式:

```python
# 随着我们不断的添加优化策略, 模型训练速度也越来越慢
# 为了能够提升fasttext模型的训练效率, 减小训练时间
# 设置train_supervised方法中的参数loss来修改损失计算方式(等效于输出层的结构), 默认是softmax层结构
# 我们这里将其设置为'hs', 代表层次softmax结构, 意味着输出层的结构(计算方式)发生了变化, 将以一种更低复杂度的方式来计算损失.
>>> model = fasttext.train_supervised(input="data/cooking/cooking.pre.train", lr=1.0, epoch=25, wordNgrams=2, loss='hs')
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread: 1341740 lr:  0.000000 avg.loss:  2.225962 ETA:   0h 0m 0s
>>> model.test("data/cooking/cooking.pre.valid")
# 我们看到精度和召回率稍有波动, 但训练时间却缩短到仅仅几秒
(3000, 0.483, 0.20887991927346114)
```



#### 7 自动超参数调优:

```python
# 手动调节和寻找超参数是非常困难的, 因为参数之间可能相关, 并且不同数据集需要的超参数也不同, 
# 因此可以使用fasttext的autotuneValidationFile参数进行自动超参数调优.
# autotuneValidationFile参数需要指定验证数据集所在路径, 它将在验证集上使用随机搜索方法寻找可能最优的超参数.
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300s, 根据不同的需求, 我们可以延长或缩短时间.
# 验证集路径'cooking.valid', 随机搜索600秒
>>> model = fasttext.train_supervised(input='data/cooking/cooking.pre.train', autotuneValidationFile='data/cooking/cooking.pre.valid', autotuneDuration=600)

Progress: 100.0% Trials:   38 Best score:  0.376170 ETA:   0h 0m 0s
Training again with best arguments
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   63791 lr:  0.000000 avg.loss:  1.888165 ETA:   0h 0m 0s
```



#### 8 实际生产中多标签多分类问题的损失计算方式

```python
# 针对多标签多分类问题, 使用'softmax'或者'hs'有时并不是最佳选择, 因为我们最终得到的应该是多个标签, 而softmax却只能最大化一个标签. 
# 所以我们往往会选择为每个标签使用独立的二分类器作为输出层结构, 
# 对应的损失计算方式为'ova'表示one vs all.
# 这种输出层的改变意味着我们在统一语料下同时训练多个二分类模型,
# 对于二分类模型来讲, lr不宜过大, 这里我们设置为0.2
>>> model = fasttext.train_supervised(input="data/cooking/cooking.pre.train", lr=0.2, epoch=25, wordNgrams=2, loss='ova')
Read 0M words
Number of words:  8952
Number of labels: 735
Progress: 100.0% words/sec/thread:   65044 lr:  0.000000 avg.loss:  7.713312 ETA:   0h 0m 0s 

# 我们使用模型进行单条样本的预测, 来看一下它的输出结果.
# 参数k代表指定模型输出多少个标签, 默认为1, 这里设置为-1, 意味着尽可能多的输出.
# 参数threshold代表显示的标签概率阈值, 设置为0.5, 意味着显示概率大于0.5的标签
>>> model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)

# 我看到根据输入文本, 输出了它的三个最有可能的标签
((u'__label__baking', u'__label__bananas', u'__label__bread'), array([1.00000, 0.939923, 0.592677]))
```




### 3.6 模型保存与重加载

```python
# 使用model的save_model方法保存模型到指定目录
# 你可以在指定目录下找到model_cooking.bin文件
>>> model.save_model("data/model/model_cooking.bin")

# 使用fasttext的load_model进行模型的重加载
>>> model = fasttext.load_model("data/model/model_cooking.bin")

# 重加载后的模型使用方法和之前完全相同
>>> model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
((u'__label__baking', u'__label__bananas', u'__label__bread'), array([1.00000, 0.939923, 0.592677]))
```




## 4 小结

* 学习了什么是文本分类:
    * 文本分类的是将文档（例如电子邮件，帖子，文本消息，产品评论等）分配给一个或多个类别. 当今文本分类的实现多是使用机器学习方法从训练数据中提取分类规则以进行分类, 因此构建文本分类器需要带标签的数据.
* 了解了Fasttext在进行文本分类时的模型架构
* 文本分类的种类:
    * 二分类:
        * 文本被分类两个类别中, 往往这两个类别是对立面, 比如: 判断一句评论是好评还是差评.
    * 单标签多分类:
        * 文本被分入到多个类别中, 且每条文本只能属于某一个类别(即被打上某一个标签), 比如: 输入一个人名, 判断它是来自哪个国家的人名.
    * 多标签多分类:
        * 文本被分人到多个类别中, 但每条文本可以属于多个类别(即被打上多个标签), 比如: 输入一段描述, 判断可能是和哪些兴趣爱好有关, 一段描述中可能即讨论了美食, 又太讨论了游戏爱好.

* 使用fasttext工具进行文本分类的过程:
    * 第一步: 获取数据
    * 第二步: 训练集与验证集的划分
    * 第三步: 训练模型
    * 第四步: 使用模型进行预测并评估
    * 第五步: 模型调优
    * 第六步: 模型保存与重加载

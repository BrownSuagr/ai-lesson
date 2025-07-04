### 学习目标

* 掌握Transformer中应用多头注意力的原因.
* 掌握Transformer中多头注意力的计算方式.



> 思考题：Transformer为什么需要进行Multi-head Attention? Multi-head Attention的计算过程是什么?



## 1 采用Multi-head Attention的原因

* 原始论文中提到进行Multi-head Attention的原因是将模型分为多个头, 可以形成多个子空间, 让模型去关注不同方面的信息, 最后再将各个方面的信息综合起来得到更好的效果.

* 多个头进行attention计算最后再综合起来, 类似于CNN中采用多个卷积核的作用, 不同的卷积核提取不同的特征, 关注不同的部分, 最后再进行融合.

* 直观上讲, 多头注意力有助于神经网络捕捉到更丰富的特征信息.



## 2 Multi-head Attention的计算方式

* Multi-head Attention和单一head的Attention唯一的区别就在于, 其对特征张量的最后一个维度进行了分割, 一般是对词嵌入的embedding_dim=512进行切割成head=8, 这样每一个head的嵌入维度就是512/8=64, 后续的Attention计算公式完全一致, 只不过是在64这个维度上进行一系列的矩阵运算而已.

* 在head=8个头上分别进行注意力规则的运算后, 简单采用拼接concat的方式对结果张量进行融合就得到了Multi-head Attention的计算结果.



## 3 小结

* 学习了Transformer架构采用Multi-head Attention的原因.
    * 将模型划分为多个头, 分别进行Attention计算, 可以形成多个子空间, 让模型去关注不同方面的信息特征, 更好的提升模型的效果.
    * 多头注意力有助于神经网络捕捉到更丰富的特征信息.

* 学习了Multi-head Attention的计算方式.
    * 对特征张量的最后一个维度进行了分割, 一般是对词嵌入的维度embedding_dim进行切割, 切割后的计算规则和单一head完全一致.
    * 在不同的head上应用了注意力计算规则后, 得到的结果张量直接采用拼接concat的方式进行融合, 就得到了Multi-head Attention的结果张量.


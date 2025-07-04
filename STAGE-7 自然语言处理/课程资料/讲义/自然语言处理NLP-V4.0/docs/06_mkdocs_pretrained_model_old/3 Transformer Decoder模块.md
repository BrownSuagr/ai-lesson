### 学习目标

* 掌握Transformer结构中的Decoder端的输入张量特点和含义.
* 掌握Decoder在训练阶段的输入是什么.
* 掌握Decoder在预测阶段的输入是什么.



> 思考题：Transformer结构中的Decoder端具体输入是什么? 在训练阶段和预测阶段一致吗?



## 1 Decoder端的输入解析

### 1.1 Decoder端的架构

Transformer原始论文中的Decoder模块是由N=6个相同的Decoder Block堆叠而成, 其中每一个Block是由3个子模块构成, 分别是多头self-attention模块, Encoder-Decoder attention模块, 前馈全连接层模块.

* 6个Block的输入不完全相同:
    * 最下面的一层Block接收的输入是经历了MASK之后的Decoder端的输入 + Encoder端的输出.
    * 其他5层Block接收的输入模式一致, 都是前一层Block的输出 + Encoder端的输出.

### 1.2 Decoder在训练阶段的输入解析

* 从第二层Block到第六层Block的输入模式一致, 无需特殊处理, 都是固定操作的循环处理.
* 聚焦在第一层的Block上: 训练阶段每一个time step的输入是上一个time step的输入加上真实标签序列向后移一位. 具体来说, 假设现在的真实标签序列等于"How are you?", 当time step=1时, 输入张量为一个特殊的token, 比如"SOS"; 当time step=2时, 输入张量为"SOS How"; 当time step=3时, 输入张量为"SOS How are", 以此类推...
* 注意: 在真实的代码实现中, 训练阶段不会这样动态输入, 而是一次性的把目标序列全部输入给第一层的Block, 然后通过多头self-attention中的MASK机制对序列进行同样的遮掩即可.



### 1.3 Decoder在预测阶段的输入解析

* 同理于训练阶段, 预测时从第二层Block到第六层Block的输入模式一致, 无需特殊处理, 都是固定操作的循环处理.
* 聚焦在第一层的Block上: 因为每一步的输入都会有Encoder的输出张量, 因此这里不做特殊讨论, 只专注于纯粹从Decoder端接收的输入. 预测阶段每一个time step的输入是从time step=0, input_tensor="SOS"开始, 一直到上一个time step的预测输出的累计拼接张量. 具体来说: 
    * 当time step=1时, 输入的input_tensor="SOS", 预测出来的输出值是output_tensor="What";
    * 当time step=2时, 输入的input_tensor="SOS What", 预测出来的输出值是output_tensor="is";
    * 当time step=3时, 输入的input_tensor="SOS What is", 预测出来的输出值是output_tensor="the";
    * 当time step=4时, 输入的input_tensor="SOS What is the", 预测出来的输出值是output_tensor="matter";
    * 当time step=5时, 输入的input_tensor="SOS What is the matter", 预测出来的输出值是output_tensor="?";
    * 当time step=6时, 输入的input_tensor="SOS What is the matter ?", 预测出来的输出值是output_tensor="EOS", 代表句子的结束符, 说明解码结束, 预测结束.




## 2 小结

* 在Transformer结构中的Decoder模块的输入, 区分于不同的Block, 最底层的Block输入有其特殊的地方. 第二层到第六层的输入一致, 都是上一层的输出和Encoder的输出.

* 最底层的Block在训练阶段, 每一个time step的输入是上一个time step的输入加上真实标签序列向后移一位. 具体来看, 就是每一个time step的输入序列会越来越长, 不断的将之前的输入融合进来.

* 最底层的Block在训练阶段, 真实的代码实现中, 采用的是MASK机制来模拟输入序列不断添加的过程.

* 最底层的Block在预测阶段, 每一个time step的输入是从time step=0开始, 一直到上一个time step的预测值的累积拼接张量. 具体来看, 也是随着每一个time step的输入序列会越来越长. 相比于训练阶段最大的不同是这里不断拼接进来的token是每一个time step的预测值, 而不是训练阶段每一个time step取得的groud truth值.




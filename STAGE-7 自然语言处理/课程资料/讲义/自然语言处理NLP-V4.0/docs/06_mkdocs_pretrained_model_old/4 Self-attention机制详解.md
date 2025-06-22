### 学习目标

* 掌握self-attention的机制和原理.
* 掌握为什么要使用三元组(Q, K, V)来计算self-attention.
* 理解softmax函数的输入是如何影响输出分布的.
* 理解softmax函数反向传播进行梯度求导的数学过程.
* 理解softmax函数出现梯度消失的原因.
* 理解self-attention计算规则中归一化的原因.



> 思考题1： Transformer中一直强调的self-attention是什么? 为什么能发挥如此大的作用? 计算的时候如果不使用三元组(Q, K, V), 而仅仅使用(Q, V)或者(K, V)或者(V)行不行?
>
> 思考题2：self-attention公式中的归一化有什么作用? 为什么要添加scaled?



## 1 Self-attention的机制和原理

self-attention是一种通过自身和自身进行关联的attention机制, 从而得到更好的representation来表达自身.

self-attention是attention机制的一种特殊情况，在self-attention中, Q=K=V, 序列中的每个单词(token)都和该序列中的其他所有单词(token)进行attention规则的计算.

attention机制计算的特点在于, 可以直接跨越一句话中不同距离的token, 可以远距离的学习到序列的知识依赖和语序结构.

<center><img src="./img/picture_8.png" height="auto" width="auto"/></center>



> * 从上图中可以看到, self-attention可以远距离的捕捉到语义层面的特征(its的指代对象是Law).

> * 应用传统的RNN, LSTM, 在获取长距离语义特征和结构特征的时候, 需要按照序列顺序依次计算, 距离越远的联系信息的损耗越大, 有效提取和捕获的可能性越小.

> * 但是应用self-attention时, 计算过程中会直接将句子中任意两个token的联系通过一个计算步骤直接联系起来, 



关于self-attention为什么要使用(Q, K, V)三元组而不是其他形式:

* 首先一条就是从分析的角度看, 查询Query是一条独立的序列信息, 通过关键词Key的提示作用, 得到最终语义的真实值Value表达, 数学意义更充分, 完备.
* 这里不使用(K, V)或者(V)没有什么必须的理由, 也没有相关的论文来严格阐述比较试验的结果差异, 所以可以作为开放性问题未来去探索, 只要明确在经典self-attention实现中用的是三元组就好.



self-attention公式中的归一化有什么作用? 为什么要添加scaled?





## 2 Self-attention中的归一化概述

* 训练上的意义: 随着词嵌入维度d_k的增大, q * k 点积后的结果也会增大, 在训练时会将带有饱和区间的激活函数（比如：sigmoid激活函数、tanh激活函数、逻辑回归softmax）推入梯度非常小的区域, 可能出现梯度消失的现象, 造成模型收敛困难.


* 数学上的意义: 假设q和k的统计变量是满足标准正态分布的独立随机变量, 意味着q和k满足均值为0, 方差为1. 那么q和k的点积结果就是均值为0, 方差为d_k, 为了抵消这种方差被放大d_k倍的影响, 在计算中主动将点积缩放1/sqrt(d_k), 这样点积后的结果依然满足均值为0, 方差为1.



## 3 softmax的梯度变化

这里我们分3个步骤来解释softmax的梯度问题:

* 第一步: softmax函数的输入分布是如何影响输出的.
* 第二步: softmax函数在反向传播的过程中是如何梯度求导的.
* 第三步: softmax函数出现梯度消失现象的原因.



### 3.1 softmax函数的输入分布是如何影响输出的

* 对于一个输入向量x, softmax函数将其做了一个归一化的映射, 首先通过自然底数e将输入元素之间的差距先"拉大", 然后再归一化为一个新的分布. 在这个过程中假设某个输入x中最大的元素下标是k, 如果输入的数量级变大(就是x中的每个分量绝对值都很大), 那么在数学上会造成y_k的值非常接近1.
* 具体用一个例子来演示, 假设输入的向量x = [a, a, 2a], 那么随便给几个不同数量级的值来看看对y3产生的影响

```text
a = 1时,   y3 = 0.5761168847658291  # e^2 / (e^1 + e^1 + e^2))
a = 10时,  y3 = 0.9999092083843412  # e^20 / (e^10 + e^10 + e^20))
a = 100时, y3 = 1.0                 # e^200 / (e^100 + e^100 + e^200))
```



> * 采用一段实例代码将a在不同取值下, 对应的y3全部画出来, 以曲线的形式展示:

```python
from math import exp
from matplotlib import pyplot as plt
import numpy as np 
f = lambda x: exp(x * 2) / (exp(x) + exp(x) + exp(x * 2))
x = np.linspace(0, 100, 100)
y_3 = [f(x_i) for x_i in x]
plt.plot(x, y_3)
plt.show()
```

> * 得到如下的曲线:

<center><img src="./img/picture_13.png" height="auto" width="auto"/></center>



> * 从上图可以很清楚的看到输入元素的数量级对softmax最终的分布影响非常之大

> * 结论： 在输入元素的数量级较大时，softmax函数几乎将全部的概率分布都分配给了最大值分量所对应的标签。通俗的讲：数据的方差变大（离散程度变大），最大值强占了所有概率。



### 3.2 softmax函数在反向传播的过程中是如何梯度求导的

softmax函数在反向传播中容易梯度消失，所以要看一看softmax函数在反向传播中是如何求导的。

首先定义神经网络的输入和输出:

> 


$$
设X=[x_1,x_2,\cdots,x_n],Y=softmax(X)=[y_1,y_2,\cdots,y_n]\\\\
则y_i=\frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}},显然\sum_{i=1}^{n}y_i=1
$$

反向传播就是输出端的损失函数对输入端求偏导的过程, 这里要分两种情况, 第一种如下所示:

$$
\begin{align*}
(1)当i=j时\\\\
\frac{\partial{y_i}}{\partial{x_j}} &= \frac{\partial{y_i}}{\partial{x_i}}\\\\
&= \frac{\partial}{\partial{x_i}}{(\frac{e^{x_i}}{\sum_k e^{x_k}})} \\\\
&= \frac{(e^{x_i})^{\prime} (\sum_k e^{x_k}) - e^{x_i}(\sum_ke^{x_k})^{\prime}}{(\sum_ke^{x_k})^2}\\\\
&=\frac{e^{x_i}\cdot(\sum_ke^{x_k})-e^{x_i}\cdot e^{x_i}}{(\sum_ke^{x_k})^2}\\\\
&=\frac{e^{x_i}\cdot(\sum_ke^{x_k})}{(\sum_ke^{x_k})^2}-\frac{e^{x_i}\cdot e^{x_i}}{(\sum_ke^{x_k})^2}\\\\
&=\frac{e^{x_i}}{\sum_ke^{x_k}}-\frac{e^{x_i}}{\sum_ke^{x_k}}\cdot \frac{ e^{x_i}}{\sum_ke^{x_k}}\\\\
&=y_i-y_i\cdot y_i \\\\
&=y_i(1-y_i) 
\end{align*}
$$

第二种如下所示:


$$
\begin{align*}
(2)当i\neq j时\\\\
\frac{\partial{y_i}}{\partial{x_j}} &= \frac{\partial}{\partial{x_j}}{(\frac{e^{x_i}}{\sum_k e^{x_k}})} \\\\
&= \frac{(e^{x_i})^{\prime} (\sum_k e^{x_k}) - e^{x_i}(\sum_ke^{x_k})^{\prime}}{(\sum_ke^{x_k})^2}\\\\
&= \frac{0\cdot (\sum_k e^{x_k}) - e^{x_i}\cdot e^{x_j}}{(\sum_ke^{x_k})^2}\\\\
&= -\frac{e^{x_i}\cdot e^{x_j}}{(\sum_ke^{x_k})^2}\\\\
&= -\frac{e^{x_i}}{\sum_ke^{x_k}}\cdot \frac{ e^{x_j}}{\sum_ke^{x_k}}\\\\
&=-y_i\cdot y_j 
\end{align*}
$$

经过对两种情况分别的求导计算, 可以得出最终的结论如下:


$$
\begin{align*}
综上所述：\frac{\partial y_i}{\partial x_j} &=  \begin{cases} y_i-y_i\cdot y_i, & \text {i=j} \\\\ 0-y_i\cdot y_j, & \text{i $\neq$ j} \end{cases} \\\\
所以：\frac{\partial Y}{\partial X} &= diag(Y)-Y^T\cdot Y \;\;\;(当Y的shape为(1,n)时)
\end{align*}
$$

把抽象的数学公式，映射成矩阵表示，见3.3节表示（i=j时，两个矩阵对角线 - 对角线 ；i!=j时，对应位置相减）。

### 3.3 softmax函数出现梯度消失现象的原因

> * 根据第二步中softmax函数的求导结果, 可以将最终的结果以矩阵形式展开如下:

$$
\frac{\partial g(X)}{\partial X}\approx \begin{bmatrix} \hat y_1 & 0 & \cdots & 0 \\ 0 & \hat y_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \hat y_d  \end{bmatrix} - \begin{bmatrix} \hat y_1^2 & \hat y_1 \hat y_2 & \cdots & \hat y_1 \hat y_d \\ \hat y_2 \hat y_1  & \hat y_2^2 & \cdots & \hat y_2 \hat y_d \\ \vdots & \vdots & \ddots & \vdots \\ \hat y_d \hat y_1 & \hat y_d \hat y_2 & \cdots & \hat y_d^2  \end{bmatrix}
$$

> * 根据第一步中的讨论结果, 当输入x的分量值较大时, softmax函数会将大部分概率分配给最大的元素, 假设最大元素是x1, 那么softmax的输出分布将产生一个接近one-hot的结果张量y_ = [1, 0, 0,..., 0], 此时结果矩阵变为:

$$
\frac{\partial g(X)}{\partial X}\approx \begin{bmatrix} 1 & 0 & \cdots & 0 \\\\ 0 & 0 & \cdots & 0 \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & 0  \end{bmatrix} - \begin{bmatrix} 1 & 0 & \cdots & 0 \\\\ 0 & 0 & \cdots & 0 \\\\ \vdots & \vdots & \ddots & \vdots \\\\ 0 & 0 & \cdots & 0  \end{bmatrix}=0
$$

> * 结论: 综上可以得出, 所有的梯度都消失为0(接近于0), 参数几乎无法更新, 模型收敛困难.




## 4 维度与点积大小的关系

* 针对为什么维度会影响点积的大小, 原始论文中有这样的一点解释如下:

```text
To illustrate why the dot products get large, assume that the components of q and k 
are independent random variables with mean 0 and variance 1. Then their doct product,
q*k = (q1k1+q2k2+......+q(d_k)k(d_k)), has mean 0 and variance d_k.
```



> * 我们分两步对其进行一个推导, 首先就是假设向量q和k的各个分量是相互独立的随机变量, X = q_i, Y = k_i, X和Y各自有d_k个分量, 也就是向量的维度等于d_k, 有E(X) = E(Y) = 0, 以及D(X) = D(Y) = 1.

> * 可以得到E(XY) = E(X)E(Y) = 0 * 0 = 0

> * 同理, 对于D(XY)推导如下:


$$
\begin{align*}
D(XY) & = E(X^2\cdot Y^2)-[E(XY)]^2 \\\\
&=E(X^2)E(Y^2)-[E(X)E(Y)]^2 \\\\
&=E(X^2-0^2)E(Y^2-0^2)-[E(X)E(Y)]^2 \\\\
&=E(X^2-[E(X)]^2)E(Y^2-[E(Y)]^2)-[E(X)E(Y)]^2 \\\\
&=D(X)D(Y)-[E(X)E(Y)]^2 \\\\
&=1 \times 1- (0 \times 0)^2 \\\\
&=1
\end{align*}
$$


> * 根据期望和方差的性质, 对于互相独立的变量满足下式:

$$
E(\sum_iZ_i) =\sum_iE(Z_i),\\\\
D(\sum_iZ_i) =\sum_iD(Z_i)
$$

> * 上述公式，简读为：和的期望，等于期望的和；和的方差等于方差的和 
> * 根据上面的公式, 可以很轻松的得出q*k的均值为E(qk) = 0, D(qk) = d_k.

> * 所以方差越大, 对应的qk的点积就越大, 这样softmax的输出分布就会更偏向最大值所在的分量.

> * 一个技巧就是将点积除以sqrt(d_k), 将方差在数学上重新"拉回1", 如下所示:

$$
D(\frac{q\cdot k}{\sqrt{d_k}})=\frac{d_k}{(\sqrt{d_k})^2}=1
$$

> * 最终的结论: 通过数学上的技巧将方差控制在1, 也就有效的控制了点积结果的发散, 也就控制了对应的梯度消失的问题!



## 5 小结

* self-attention机制的重点是使用三元组(Q, K, V)参与规则运算, 这里面Q=K=V.
* self-attention最大的优势是可以方便有效的提取远距离依赖的特征和结构信息, 不必向RNN那样依次计算产生传递损耗.
* 关于self-attention采用三元组的原因, 经典实现的方式数学意义明确, 理由充分, 至于其他方式的可行性暂时没有论文做充分的对比试验研究.
* 学习了softmax函数的输入是如何影响输出分布的.
    * softmax函数本质是对输入的数据分布做一次归一化处理, 但是输入元素的数量级对softmax最终的分布影响非常之大.
    * 在输入元素的数量级较大时, softmax函数几乎将全部的概率分布都分配给了最大值分量所对应的标签.
* 学习了softmax函数在反向传播的过程中是如何梯度求导的.
    * 具体的推导过程见讲义正文部分, 注意要分两种情况讨论, 分别处理.

* 学习了softmax函数出现梯度消失现象的原因.
    * 结合第一步, 第二步的结论, 可以很清楚的看到最终的梯度矩阵接近于零矩阵, 这样在进行参数更新的时候就会产生梯度消失现象.

* 学习了维度和点积大小的关系推导.
    * 通过期望和方差的推导理解了为什么点积会造成方差变大.
    * 理解了通过数学技巧除以sqrt(d_k)就可以让方差恢复成1.






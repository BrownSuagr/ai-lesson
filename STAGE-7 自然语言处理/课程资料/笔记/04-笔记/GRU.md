# RNN及其变体

### GRU模型

- 内部结构
  - 更新门
  - 重置门
- GRU模型：

![image-20240618174404615](img/06.png)

- GRU模型代码实现

  ```python
  # coding:utf-8
  import torch
  import torch.nn as nn
  
  # 模型参数发生变化对其他输入参数的影响
  def  dm01_gru_():
      '''
      第一个参数：input_size(输入张量x的维度)
      第二个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
      第三个参数：num_layer(隐藏层的数量)
      # batch_first = True，代表batch_size 放在第一位
      '''
      gru = nn.GRU(5, 6, 1, batch_first=True)
      print(gru.all_weights)
      print(gru.all_weights[0][0].shape)
      print(gru.all_weights[0][1].shape)
  
      '''
      第一个参数：batch_size(批次的样本数量)
      第二个参数：sequence_length(输入序列的长度)
      第三个参数：input_size(输入张量的维度)
      '''
      input = torch.randn(4, 3, 5)
  
      '''
      第一个参数：num_layer * num_directions(层数*网络方向)
      第二个参数：batch_size(批次的样本数)
      第三个参数：hidden_size(隐藏层的维度， 隐藏层神经元的个数)
      '''
  
      h0 = torch.randn(1, 4, 6)
  
      # 将数据送入模型得到结果
      output, hn = gru(input, h0)
      print(f'output--》{output}')
      print(f'hn--》{hn}')
  
  if __name__ == '__main__':
      dm01_gru_()
  ```

- BI-GRU

```properties
定义: 不改变原始的GRU模型内部结构，只是将文本从左到右计算一遍，再从右到左计算一遍，把最终的输出结果拼接得到模型的完整输出
```

- 优缺点
  - 优点：相比LSTM，结构较为简单，能够和lstm一样缓解梯度消失问题
  - 缺点：RNN系列模型不能实现并行运算，数据量大的话，效率比较低


















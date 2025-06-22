# coding:utf-8
import torch.nn as nn
import torch
torch.manual_seed(8)
def dm01_rnn_base():
    # 举例：每个样本只有1个单词
    # 1.实例化RNN模型
    # input_size--》代表input中每个单词的词嵌入维度
    # hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # num_layers--》代表RNN单元个数（几层隐藏层）
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)
    # 2.指定input输入
    # 默认RNN要求输入的形状是【sequence_length, batch_size, embed_dim】
    # 但是，我们最常见模型输入形状--》[batch_size, sequence_length, embed_dim]
    # 第一个参数：sequence_length每个样本的句子长度
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数：embed_dim 每个单词词嵌入的维度
    input = torch.randn(1, 3, 5)
    # 3.指定输入h0
    # 第一个参数：num_layers代表RNN单元个数（几层隐藏层）
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数 hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    h0 = torch.randn(1, 3, 6)
    # 4.将input和ho送入模型
    output, hn = rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')

# 改变句子长度
def dm02_rnn_seqLen():
    # 举例：每个样本只有1个单词
    # 1.实例化RNN模型
    # input_size--》代表input中每个单词的词嵌入维度
    # hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # num_layers--》代表RNN单元个数（几层隐藏层）
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)
    # 2.指定input输入
    # 默认RNN要求输入的形状是【sequence_length, batch_size, embed_dim】
    # 但是，我们最常见模型输入形状--》[batch_size, sequence_length, embed_dim]
    # 第一个参数：sequence_length每个样本的句子长度
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数：embed_dim 每个单词词嵌入的维度
    input = torch.randn(4, 3, 5)
    # 3.指定输入h0
    # 第一个参数：num_layers代表RNN单元个数（几层隐藏层）
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数 hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    h0 = torch.randn(1, 3, 6)
    # 4.将input和ho送入模型
    # output-->保存每个时间步（单词）输出的隐藏层结果，hn只是代表最后一个单词的输出结果
    output, hn = rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')
# RNN模型循环机制的理解：1个样本
def dm03_rnn_for_multinum():
    # 构建模型
    rnn = nn.RNN(5, 6, 1)
    # 定义输入
    input = torch.randn(4, 1, 5)
    hidden = torch.zeros(1, 1, 6)
    # 方式1：1个1个的送字符
    # for i in range(4):
    #     tmp = input[i][0]
    #     print("tmp.shape--->", tmp.shape) # 拿出1个数组
    #     output, hidden = rnn(tmp.unsqueeze(0).unsqueeze(0), hidden)
    #     print(i+1,'output--->', output, )
    #     print(i+1,'hidden--->', hidden, )
    #     print('*'*80)
    # # 一次性将数据送入模型
    hidden = torch.zeros(1, 1, 6)
    output, hn = rnn(input, hidden)
    print('output2--->', output, output.shape)
    print('hn--->', hn, hn.shape)

# RNN模型循环机制的理解:多个样本
def dm04_rnn_for_multinum():
    # 构建模型
    rnn = nn.RNN(5, 6, 1)
    # 定义输入
    input = torch.randn(4, 3, 5)
    hidden = torch.zeros(1, 3, 6)
    # 方式1：1个1个的送字符
    for i in range(4):
        tmp = input[i, :, :]
        print("tmp.shape--->", tmp.shape) # 拿出1个数组
        output, hidden = rnn(tmp.unsqueeze(0), hidden)
        print(i+1,'output--->', output, )
        print(i+1,'hidden--->', hidden, )
        print('*'*80)
    # # 一次性将数据送入模型
    hidden = torch.zeros(1, 3, 6)
    output, hn = rnn(input, hidden)
    print('output2--->', output, output.shape)
    print('hn--->', hn, hn.shape)

# 增加N层隐藏层
def dm05_rnn_numlayers():
    # 举例：每个样本只有1个单词
    # 1.实例化RNN模型
    # input_size--》代表input中每个单词的词嵌入维度
    # hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # num_layers--》代表RNN单元个数（几层隐藏层）
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=2)
    # 2.指定input输入
    # 默认RNN要求输入的形状是【sequence_length, batch_size, embed_dim】
    # 但是，我们最常见模型输入形状--》[batch_size, sequence_length, embed_dim]
    # 第一个参数：sequence_length每个样本的句子长度
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数：embed_dim 每个单词词嵌入的维度
    input = torch.randn(4, 3, 5)
    # 3.指定输入h0
    # 第一个参数：num_layers代表RNN单元个数（几层隐藏层）
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数 hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    h0 = torch.randn(2, 3, 6)
    # 4.将input和ho送入模型
    output, hn = rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')


# 改变batch_first参数
def dm06_rnn_batchFirst():

    # 1.实例化RNN模型
    # input_size--》代表input中每个单词的词嵌入维度
    # hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # num_layers--》代表RNN单元个数（几层隐藏层）
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1, batch_first=True)
    # 2.指定input输入
    # 默认RNN要求输入的形状是【sequence_length, batch_size, embed_dim】
    # 但是，如果改变了batch_first=True，模型输入形状--》[batch_size, sequence_length, embed_dim]
    # 第一个参数：batch_size 一个批次几个样本
    # 第二个参数：sequence_length每个样本的句子长度
    # 第三个参数：embed_dim 每个单词词嵌入的维度
    input = torch.randn(4, 3, 5)
    # 3.指定输入h0
    # 第一个参数：num_layers代表RNN单元个数（几层隐藏层）
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数 hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    h0 = torch.randn(1, 4, 6)
    # 4.将input和ho送入模型
    output, hn = rnn(input, h0)
    print(f'output--》{output}')
    print(f'hn--》{hn}')
if __name__ == '__main__':
    # dm01_rnn_base()
    # dm02_rnn_seqLen()
    # dm03_rnn_for_multinum()
    # dm04_rnn_for_multinum()
    # dm05_rnn_numlayers()
    dm06_rnn_batchFirst()
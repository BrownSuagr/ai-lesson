import torch.nn as nn
import torch

def dm01_lstm():
    # 1.实例化LSTM模型
    # input_size--》代表input中每个单词的词嵌入维度
    # hidden_size--》代表指定LSTM模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # num_layers--》代表LSTM单元个数
    # lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1)
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, batch_first=True)
    # 2.指定input输入
    # 默认LSTM要求输入的形状是【sequence_length, batch_size, embed_dim】
    # 但是，我们最常见模型输入形状--》[batch_size, sequence_length, embed_dim]
    # 第一个参数：sequence_length每个样本的句子长度
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数：embed_dim 每个单词词嵌入的维度
    input = torch.randn(4, 3, 5)
    # 3.指定输入h0
    # 第一个参数：num_layers代表RNN单元个数（几层隐藏层）
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数 hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # h0 = torch.randn(1, 3, 6)
    #  batch_first=True
    h0 = torch.randn(1, 4, 6)
    # 4.指定输入c0
    # 第一个参数：num_layers代表RNN单元个数（几层隐藏层）
    # 第二个参数：batch_size 一个批次几个样本
    # 第三个参数 hidden_size--》代表指定RNN模型输出的维度（隐藏层维度//隐藏层神经元个数）
    # c0 = torch.randn(1, 3, 6)
    #  batch_first=True
    c0 = torch.randn(1, 4, 6)

    # 4.将input和ho,co送入模型
    output, (hn, cn) = lstm(input, (h0, c0))
    print(f'output--》{output}')
    print(f'hn--》{hn}')
    print(f'cn--》{cn}')

if __name__ == '__main__':
    dm01_lstm()
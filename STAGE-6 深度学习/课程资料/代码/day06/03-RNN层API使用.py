import torch
import torch.nn as nn


def dm01():
	# 创建RNN层
	# input_size: 词向量维度
	# hidden_size: 隐藏状态向量维度
	# num_layers: 隐藏层数量, 默认1层
	rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
	# 输入x
	# (5, 32, 128)->(每个句子的词个数, 句子数, 词向量维度)
	# RNN对象input_size=词向量维度
	x = torch.randn(size=(5, 32, 128))
	# 上一个时间步的隐藏状态h0
	# (1,32,256->(隐藏层数量, 句子数, 隐藏状态向量维度)
	# RNN对象hidden_size=隐藏状态向量维度
	h0 = torch.randn(size=(1, 32, 256))
	# 调用RNN层输出当前预测值和当前的隐藏状态h1
	output, h1 = rnn(x, h0)
	print(output.shape, h1.shape)


if __name__ == '__main__':
	dm01()

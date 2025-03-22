import torch
import torch.nn as nn
import math


# 随机参数初始化
def dm01():
	# 创建线性层对象, 对线性层的权重进行初始化
	# in_features: 输入神经元个数
	# out_features: 输出神经元个数
	linear1 = nn.Linear(in_features=5, out_features=8)
	linear2 = nn.Linear(in_features=8, out_features=10)
	# 均匀分布初始化
	nn.init.uniform_(linear1.weight)
	nn.init.uniform_(linear1.weight, a=-1/torch.sqrt(torch.tensor(5.0)), b=1/torch.sqrt(torch.tensor(5.0)))
	nn.init.uniform_(linear1.bias)
	print(linear1.weight)
	print(linear1.bias)


# 正态分布参数初始化
def dm02():
	# 创建线性层对象, 对线性层的权重进行初始化
	# in_features: 输入神经元个数
	# out_features: 输出神经元个数
	linear1 = nn.Linear(in_features=5, out_features=8)
	linear2 = nn.Linear(in_features=8, out_features=10)
	# 均匀分布初始化
	nn.init.normal_(linear1.weight)
	nn.init.normal_(linear1.bias)
	print(linear1.weight)
	print(linear1.bias)


# nn.init.zeros_()  # 全0初始化
# nn.init.ones_()  # 全1初始化
# nn.init.constant_(val=0.1)  # 全固定值初始化
# nn.init.kaiming_uniform_()  # 凯明均匀分布初始化
# nn.init.kaiming_normal_()  # 凯明正态分布初始化
# nn.init.xavier_uniform_()  # xavier均匀分布初始化
# nn.init.xavier_normal_()  # xavier正态分布初始化


if __name__ == '__main__':
	dm01()
	dm02()

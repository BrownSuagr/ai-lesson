import torch
import pandas as pd


def dm01():
	# 创建输出层加权求和值
	y = torch.tensor(data=[[0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75],
						   [0.2, 0.02, 0.15, 3.75, 1.3, 0.5, 0.06, 1.1, 0.05, 0.15]])
	# softmax激活函数转换成概率值
	# 1轴按列计算
	# y_softmax = torch.softmax(input=y, dim=-1)
	y_softmax = torch.softmax(input=y, dim=1)
	print('y_softmax->', y_softmax)


if __name__ == '__main__':
	dm01()

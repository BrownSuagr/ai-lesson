# 动量法计算梯度实际上计算的是当前时刻的指数移动加权平均梯度值
import torch
from torch import optim


def dm01():
	# todo: 1-初始化权重参数
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
	loss = ((w ** 2) / 2.0).sum()
	# todo: 2-创建优化器函数对象 SGD->动量法
	# momentum: 动量法, 一般0.9或0.99
	optimizer = optim.SGD([w], lr=0.01, momentum=0.9)
	# todo: 3-计算梯度值
	optimizer.zero_grad()
	loss.sum().backward()
	# todo: 4-更新权重参数 梯度更新
	optimizer.step()
	print('w.grad->', w.grad, w)
	# 第二次计算
	loss = ((w ** 2) / 2.0).sum()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('w.grad->', w.grad, w)


if __name__ == '__main__':
	dm01()

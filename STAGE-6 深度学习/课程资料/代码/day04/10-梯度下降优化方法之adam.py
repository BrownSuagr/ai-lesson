# adam既优化学习率又优化梯度  adam=rmsprop+momentum
import torch
from torch import optim


def dm01():
	# todo: 1-初始化权重参数
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
	loss = ((w ** 2) / 2.0).sum()
	# todo: 2-创建优化器函数对象
	# betas: 两个β值, 接收元组类型
	optimizer = optim.Adam([w], lr=0.01, betas=(0.9, 0.99))
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
	# 第三次计算
	loss = ((w ** 2) / 2.0).sum()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('w.grad->', w.grad, w)


if __name__ == '__main__':
	dm01()

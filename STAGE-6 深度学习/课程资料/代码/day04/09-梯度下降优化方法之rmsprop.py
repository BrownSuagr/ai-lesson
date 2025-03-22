# rmsprop优化方法调整学习率, 对adagrad方法进行优化, 使用指数移动加权平均梯度代替历史累加梯度
# 避免学习率过快减小, 导致后续模型收敛速度慢
import torch
from torch import optim


def dm01():
	# todo: 1-初始化权重参数
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
	loss = ((w ** 2) / 2.0).sum()
	# todo: 2-创建优化器函数对象  RMSprop
	optimizer = optim.RMSprop([w], lr=0.01, alpha=0.9)
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

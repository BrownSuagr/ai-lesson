import torch
import torch.nn as nn  # 损失函数,优化器函数,模型函数


def dm01():
	# todo:1-定义样本的x和y
	x = torch.ones(size=(2, 5))
	y = torch.zeros(size=(2, 3))
	print('x->', x)
	print('y->', y)
	# todo:2-初始模型权重 w b 自动微分张量
	w = torch.randn(size=(5, 3), requires_grad=True)
	b = torch.randn(size=(3,), requires_grad=True)
	print('w->', w)
	print('b->', b)
	# todo:3-初始模型 线性回归模型,计算预测y值
	y_pred = torch.matmul(x, w) + b
	print('y_pred->', y_pred)
	# todo:4-根据MSE损失函数计算损失值
	# 创建MSE对象, 类创建对象
	criterion = nn.MSELoss()
	loss = criterion(y_pred, y)
	print('loss->', loss)
	# todo:5-反向传播,计算w和b梯度
	loss.sum().backward()
	print('w.grad->', w.grad)
	print('b.grad->', b.grad)


if __name__ == '__main__':
	dm01()

# 适用于二分类
import torch
import torch.nn as nn


def dm01():
	# 手动创建样本的真实y值
	y_true = torch.tensor(data=[0, 1, 0], dtype=torch.float32)
	print('y_true->', y_true.dtype)
	# 手动创建样本的预测y值 -> 模型预测值
	# 0.6901, 0.5459, 0.2469 -> sigmoid函数的激活值
	y_pred = torch.tensor(data=[0.6901, 0.5459, 0.2469], requires_grad=True, dtype=torch.float32)
	# 创建多分类交叉熵损失对象
	# reduction:损失值计算的方式, 默认mean 平均损失值
	criterion = nn.BCELoss()
	# 调用损失对象计算损失值
	# 预测y  真实y
	loss = criterion(y_pred, y_true)
	print('loss->', loss)
	print('loss->', loss.requires_grad)


if __name__ == '__main__':
	dm01()

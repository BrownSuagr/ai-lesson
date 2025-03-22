# 适用于多分类
import torch
import torch.nn as nn


def dm01():
	# 手动创建样本的真实y值
	# y_true = torch.tensor(data=[[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
	y_true = torch.tensor(data=[1, 2])
	print('y_true->', y_true.dtype)
	# 手动创建样本的预测y值 -> 模型预测值
	y_pred = torch.tensor(data=[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], requires_grad=True, dtype=torch.float32)
	# 创建多分类交叉熵损失对象
	# reduction:损失值计算的方式, 默认mean 平均损失值
	criterion = nn.CrossEntropyLoss(reduction='sum')
	# 调用损失对象计算损失值
	# 预测y  真实y
	loss = criterion(y_pred, y_true)
	print('loss->', loss)


if __name__ == '__main__':
	dm01()

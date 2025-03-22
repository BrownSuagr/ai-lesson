# 适用于回归任务, 不受异常值影响
# smooth l1: 任意位置都可导, 越接近底部, 导数越小
import torch
import torch.nn as nn


def dm01():
	# 手动创建样本的真实y值
	y_true = torch.tensor(data=[1.2, 1.5, 2.0], dtype=torch.float32)
	print('y_true->', y_true.dtype)
	# 手动创建样本的预测y值 -> 模型预测值
	# 0.6901, 0.5459, 0.2469 -> sigmoid函数的激活值
	y_pred = torch.tensor(data=[1.3, 1.7, 2.0], requires_grad=True, dtype=torch.float32)
	# 创建回归任务smoothl1损失对象
	# reduction:损失值计算的方式, 默认mean 平均损失值
	criterion = nn.SmoothL1Loss()
	# 调用损失对象计算损失值
	# 预测y  真实y
	loss = criterion(y_pred, y_true)
	print('loss->', loss)
	print('loss->', loss.requires_grad)


if __name__ == '__main__':
	dm01()

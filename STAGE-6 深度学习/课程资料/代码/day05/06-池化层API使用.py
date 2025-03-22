import torch
import torch.nn as nn
"""
最大池化
kernel_size:窗口形状大小, 不是神经元形状大小, 池化层没有神经元参与
nn.MaxPool2d(kernel_size=, stride=, padding=)
平均池化
nn.AVGPool2d(kernel_size=, stride=, padding=)
"""

# 单通道卷积层特征图池化
def dm01():
	# 创建1通道的3*3二维矩阵, 一张特征图
	inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=torch.float)
	print('inputs->', inputs)
	print('inputs.shape->', inputs.shape)
	# 创建池化层
	# kernel_size: 窗口的形状大小
	pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool1(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)
	pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool2(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)


# 多通道卷积层特征图池化
def dm02():
	# size(3,3,3)
	inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
						   [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
						   [[11, 22, 33], [44, 55, 66], [77, 88, 99]]], dtype=torch.float)
	# 创建池化层
	# kernel_size: 窗口的形状大小
	pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool1(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)
	pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool2(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)


if __name__ == '__main__':
	# dm01()
	dm02()

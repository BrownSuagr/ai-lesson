"""
正则化: 每批样本的均值和方差不一样, 引入噪声样本
加快模型收敛: 样本标准化后, 落入激活函数的合理区间, 导数尽可能最大
"""
import torch
import torch.nn as nn


# nn.BatchNorm1d(): 处理一维样本, 每批样本数最少是2个, 否则无法计算均值和标准差
# nn.BatchNorm2d(): 处理二维样本, 图像(每个通道由二维矩阵组成), 计算二维矩阵每列均值和标准差
# nn.BatchNorm3d(): 处理三维样本, 视频
# 处理二维数据
def dm01():
	# todo:1-创建图像样本数据集 2个通道,每个通道3*4列特征图, 卷积层处理的特征图样本
	# 数据集只有一张图像, 图像是由2个通道组成, 每个通道由3*4像素矩阵
	input_2d = torch.randn(size=(1, 2, 3, 4))
	print('input_2d->', input_2d)
	# todo:2-创建BN层, 标准化 ->一定是在激活函数前进行标准化
	# num_features: 输入样本的通道数
	# eps: 小常数, 避免除0
	# momentum: 指数移动加权平均值
	# affine: 默认True, 引入可学习的γ和β参数
	bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)
	ouput_2d = bn2d(input_2d)
	print('ouput_2d->', ouput_2d)

# 处理一维数据
def dm02():
	# 创建样本数据集
	input_1d = torch.randn(size=(2, 2))
	# 创建线性层
	linear1 = nn.Linear(in_features=2, out_features=4)
	l1 = linear1(input_1d)
	print('l1->', l1)
	# 创建BN层
	bn1d = nn.BatchNorm1d(num_features=4)
	# 对线性层的结果进行标准化处理
	output_1d = bn1d(l1)
	print('output_1d->', output_1d)



if __name__ == '__main__':
	# dm01()
	dm02()

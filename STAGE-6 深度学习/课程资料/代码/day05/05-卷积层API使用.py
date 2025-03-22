import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
in_channels:原图像的通道数,RGB彩色图像是3
out_channels:卷积核/神经元个数 输出的新图像是由n个通道的二维矩阵组成
kernel_size:卷积核形状 (3,3) (3,5)
stride:步长 默认为1
padding:填充圈数 默认为0  1  same->stride=1  2,3...
nn.Conv2d(in_channels=,out_channels=,kernel_size=,stride=,padding=)
"""


def dm01():
	# todo:1-加载RGB彩色图像 (H,W,C)
	img = plt.imread(fname='data/img.jpg')
	print('img->', img)
	print('img.shape->', img.shape)
	# todo:2-将图像的形状(H,W,C)转换成(C,H,W)  permute()方法
	img2 = torch.tensor(data=img, dtype=torch.float32).permute(dims=(2, 0, 1))
	print('img2->', img2)
	print('img2.shape->', img2.shape)
	# todo:3-将这张图像保存到数据集中 (batch_size,C,H,W)  unsqueeze()方法
	# 数据集只有一个样本
	img3 = img2.unsqueeze(dim=0)
	print('img3->', img3)
	print('img3.shape->', img3.shape)
	# todo:4-创建卷积层对象, 提取特征图
	conv = nn.Conv2d(in_channels=3,
					 out_channels=4,
					 kernel_size=(3, 3),
					 stride=2,
					 padding=0)
	conv_img = conv(img3)
	print('conv_img->', conv_img)
	print('conv_img.shape->', conv_img.shape)

	# 查看提取到的4个特征图
	# 获取数据集中第一张图像
	img4 = conv_img[0]
	# 转换形状 (H,W,C)
	img5 = img4.permute(1, 2, 0)
	print('img5->', img5)
	print('img5.shape->', img5.shape)
	# img5->(H,W,C)
	# img5[:, :, 0]->第1个通道的二维矩阵特征图
	feature1 = img5[:, :, 0].detach().numpy()
	plt.imshow(feature1)
	plt.show()


if __name__ == '__main__':
	dm01()

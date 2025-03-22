import numpy as np
import matplotlib.pyplot as plt
import torch


# 创建全黑和全白图片
def dm01():
	# 全黑图片
	# 创建3通道二维矩阵, 黑色 0像素点
	# H W C: 200, 200, 3
	# 高 宽 通道
	img1 = np.zeros(shape=(200, 200, 3))
	print('img1->', img1)
	print('img1.shape->', img1.shape)
	# 展示图像
	plt.imshow(img1)
	plt.show()

	# 全白图片
	img2 = torch.full(size=(200, 200, 3), fill_value=255)
	print('img2->', img2)
	print('img2.shape->', img2.shape)
	# 展示图像
	plt.imshow(img2)
	plt.show()


def dm02():
	# 加载图片
	img1 = plt.imread(fname='data/img.jpg')
	print('img1->', img1)
	print('img1.shape->', img1.shape)
	# 保存图像
	plt.imsave(fname='data/img1.png', arr=img1)
	# 展示图像
	plt.imshow(img1)
	plt.show()


if __name__ == '__main__':
	# dm01()
	dm02()

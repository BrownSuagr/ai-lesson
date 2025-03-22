import torch


# 点乘: 又称为阿达玛积, 张量元素级乘法, 对应位置的元素进行点乘, 一般要求两个张量形状相同  *  mul()
def dm01():
	# t1 = torch.tensor(data=[[1, 2], [3, 4]])
	# (2, )
	t1 = torch.tensor(data=[1, 2])
	# (2, 2)
	t2 = torch.tensor(data=[[5, 6], [7, 8]])
	t3 = t1 * t2
	print('t3->', t3)
	t4 = torch.mul(input=t1, other=t2)
	print('t4->', t4)


if __name__ == '__main__':
	dm01()

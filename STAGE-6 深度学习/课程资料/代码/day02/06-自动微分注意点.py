# 自动微分的张量不能转换成numpy数组, 可以借助detach()方法生成新的不自动微分张量
import torch


def dm01():
	x1 = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
	print('x1->', x1)
	# 判断张量是否自动微分 返回True/False
	print(x1.requires_grad)
	# 调用detach()方法对x1进行剥离, 得到新的张量,不能自动微分,数据和原张量共享
	x2 = x1.detach()
	print(x2.requires_grad)
	print(x1.data)
	print(x2.data)
	print(id(x1.data))
	print(id(x2.data))
	# 自动微分张量转换成numpy数组
	n1 = x2.numpy()
	print('n1->', n1)


if __name__ == '__main__':
	dm01()

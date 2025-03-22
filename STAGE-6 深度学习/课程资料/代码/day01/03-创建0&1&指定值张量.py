import torch


# torch.ones(size=): 根据形状创建全1张量
# torch.ones_like(input=): 根据指定张量的形状创建全1张量
def dm01():
	t1 = torch.ones(size=(2, 3))
	print('t1的值是->', t1)
	print('t1的形状是->', t1.shape)
	print('t1的元素类型是->', t1.dtype)
	# 形状: (5, )
	t2 = torch.tensor(data=[1, 2, 3, 4, 5])
	t3 = torch.ones_like(input=t2)
	print('t2的形状是->', t2.shape)
	print('t3的值是->', t3)
	print('t3的形状是->', t3.shape)


# torch.zeros(size=): 根据形状创建全0张量
# torch.zeros_like(input=): 根据指定张量的形状创建全0张量
def dm02():
	t1 = torch.zeros(size=(2, 3))
	print('t1的值是->', t1)
	print('t1的形状是->', t1.shape)
	print('t1的元素类型是->', t1.dtype)
	# 形状: (5, )
	t2 = torch.tensor(data=[1, 2, 3, 4, 5])
	t3 = torch.zeros_like(input=t2)
	print('t2的形状是->', t2.shape)
	print('t3的值是->', t3)
	print('t3的形状是->', t3.shape)


# torch.full(size=, fill_value=): 根据形状和指定值创建指定值的张量
# torch.full_like(input=, fill_value=): 根据指定张量形状和指定值创建指定值的张量
def dm03():
	t1 = torch.full(size=(2, 3, 4), fill_value=10)
	t2 = torch.tensor(data=[[1, 2], [3, 4]])
	t3 = torch.full_like(input=t2, fill_value=100)
	print('t1的值是->', t1)
	print('t1的形状是->', t1.shape)
	print('t3的值是->', t3)
	print('t3的形状是->', t3.shape)


if __name__ == '__main__':
	# dm01()
	# dm02()
	dm03()

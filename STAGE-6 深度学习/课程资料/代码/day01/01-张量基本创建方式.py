import torch
import numpy as np


# torch.tensor(data=, dtype=,): 根据指定数据或指定元素类型创建张量
# data: 数据
# dtype: 元素类型
def dm01():
	list1 = [[1., 2, 3], [4, 5, 6]]
	int1 = 10
	# array默认类型是float64, 所以创建的张量为float64
	n1 = np.array([[1., 2., 3.], [4., 5., 6.]])
	t1 = torch.tensor(data=list1)
	t2 = torch.tensor(data=int1)
	t3 = torch.tensor(data=n1)
	# t4 = t3.to('cuda:0')  # 将cpu张量移动到gpu
	print('t1的值->', t1)
	print('t1类型->', type(t1))
	print('t1元素类型->', t1.dtype)
	print('t2的值->', t2)
	print('t2类型->', type(t2))
	print('t3的值->', t3)
	print('t3类型->', type(t3))


# torch.Tensor(data=, size=): 根据指定数据或指定形状创建张量
# data: 数据
# size: 形状, 接收元组 (0轴, 1轴, ...) -> 元组有多少个元素就是多少维张量, 对应维度上值就是数据个数
def dm02():
	# 指定数据
	t1 = torch.Tensor(data=[[1.1, 1.2, 1.3], [2.2, 2.3, 2.4]])
	print('t1的值->', t1)
	print('t1类型->', type(t1))
	# 指定形状
	t2 = torch.Tensor(size=(2, 3))
	print('t2的值->', t2)
	print('t2类型->', type(t2))


# torch.IntTensor(data=)/LongTensor()/FloatTensor()/DoubleTensor(): 创建指定类型的张量
# data: 数据
def dm03():
	# 如果元素类型不是指定类型, 会自动转换
	t1 = torch.IntTensor([[1.1, 2, 3.7], [4, 5, 6]])
	t2 = torch.FloatTensor([[1.1, 2, 3.7], [4, 5, 6]])
	print('t1的值->', t1)
	print('t1类型->', type(t1))
	print('t1元素类型->', t1.dtype)
	print('t2的值->', t2)
	print('t2类型->', type(t2))
	print('t2元素类型->', t2.dtype)


if __name__ == '__main__':
	# dm01()
	# dm02()
	dm03()

#%%

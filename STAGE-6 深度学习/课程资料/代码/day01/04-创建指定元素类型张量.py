import torch


# torch.tensor(data=, dtype=):
# dtype: 指定元素类型, 浮点类型默认是float32

# tensor.type(dtype=): 修改张量元素类型
# torch.float32
# torch.FloatTensor
# torch.cuda.FloatTensor
def dm01():
	t1 = torch.tensor(data=[[1., 2., 3.], [4., 5., 6.]], dtype=torch.float16)
	print('t1的元素类型->', t1.dtype)
	# 转换成float32
	t2 = t1.type(dtype=torch.FloatTensor)
	t3 = t1.type(dtype=torch.int64)
	print('t2的元素类型->', t2.dtype)
	print('t3的元素类型->', t3.dtype)


# tensor.half()/float()/double()/short()/int()/long()
def dm02():
	t1 = torch.tensor(data=[1, 2])
	print('t1的元素类型->', t1.dtype)
	# t2 = t1.half()
	t2 = t1.int()
	print(t2)
	print('t2的元素类型->', t2.dtype)


if __name__ == '__main__':
	# dm01()
	dm02()

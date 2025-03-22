import torch


# torch.arange(start=, end=, step=): 创建指定步长的线性张量  左闭右开
# start: 起始值
# end: 结束值
# step: 步长, 默认1
# torch.linspace(start=, end=, steps=): 创建指定元素个数的线性张量  左闭右闭
# start: 起始值
# end: 结束值
# steps: 元素个数
# step=(end-start)/(steps-1)  value_i=start+step*i
def dm01():
	t1 = torch.arange(start=0, end=10, step=2)
	print('t1的值是->', t1)
	print('t1类型是->', type(t1))
	t2 = torch.linspace(start=0, end=9, steps=9)
	print('t2的值是->', t2)
	print('t2类型是->', type(t2))


# torch.rand(size=)/randn(size=): 创建指定形状的随机浮点类型张量
# torch.randint(low=, high=, size=): 创建指定形状指定范围随机整数类型张量  左闭右开
# low: 最小值
# high: 最大值
# size: 形状, 元组

# torch.initial_seed(): 查看随机种子数
# torch.manual_seed(seed=): 设置随机种子数
def dm02():
	# (5, 4): 5行4列
	t1 = torch.rand(size=(5, 4))
	print('t1的值是->', t1)
	print('t1类型->', type(t1))
	print('t1元素类型->', t1.dtype)
	print('t1随机种子数->', torch.initial_seed())
	# 设置随机种子数
	torch.manual_seed(seed=66)
	t2 = torch.randint(low=0, high=10, size=(2, 3))
	print('t2的值是->', t2)
	print('t2类型->', type(t2))
	print('t2元素类型->', t2.dtype)
	print('t2随机种子数->', torch.initial_seed())


if __name__ == '__main__':
	# dm01()
	dm02()

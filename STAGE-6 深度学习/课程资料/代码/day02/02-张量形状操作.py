import torch


# reshape(shape=(行,列)): 修改连续或非连续张量的形状, 不改数据
# -1: 表示自动计算行或列   例如:  (5, 6) -> (-1, 3) -1*3=5*6 -1=10  (10, 3)
def dm01():
	torch.manual_seed(0)
	t1 = torch.randint(0, 10, (5, 6))
	print('t1->', t1)
	print('t1的形状->', t1.shape)
	# 形状修改为 (2, 15)
	t2 = t1.reshape(shape=(2, 15))
	t3 = t1.reshape(shape=(2, -1))
	print('t2->', t2)
	print('t2的形状->', t2.shape)
	print('t3->', t3)
	print('t3的形状->', t3.shape)


# squeeze(dim=): 删除值为1的维度, dim->指定维度, 维度值不为1不生效  不设置dim,删除所有值为1的维度
# 例如: (3,1,2,1) -> squeeze()->(3,2)  squeeze(dim=1)->(3,2,1)
# unqueeze(dim=): 在指定维度上增加值为1的维度  dim=-1:最后维度
def dm02():
	torch.manual_seed(0)
	# 四维
	t1 = torch.randint(0, 10, size=(3, 1, 2, 1))
	print('t1->', t1)
	print('t1的形状->', t1.shape)
	# squeeze: 降维
	t2 = torch.squeeze(t1)
	print('t2->', t2)
	print('t2的形状->', t2.shape)
	# dim: 指定维度
	t3 = torch.squeeze(t1, dim=1)
	print('t3->', t3)
	print('t3的形状->', t3.shape)
	# unsqueeze: 升维
	# (3, 2)->(1, 3, 2)
	# t4 = t2.unsqueeze(dim=0)
	# 最后维度 (3, 2)->(3, 2, 1)
	t4 = t2.unsqueeze(dim=-1)
	print('t4->', t4)
	print('t4的形状->', t4.shape)


# 调换维度
# torch.permute(input=,dims=): 改变张量任意维度顺序
# input: 张量对象
# dims: 改变后的维度顺序, 传入轴下标值 (1,2,3)->(3,1,2)
# torch.transpose(input=,dim0=,dim1=): 改变张量两个维度顺序
# dim0: 第一个维度
# dim1: 第二个维度
# (1,2,3)->(2,1,3) 一次只能交换两个维度
def dm03():
	torch.manual_seed(0)
	t1 = torch.randint(low=0, high=10, size=(3, 4, 5))
	print('t1->', t1)
	print('t1形状->', t1.shape)
	# 交换0维和1维数据
	# t2 = t1.transpose(dim0=1, dim1=0)
	t2 = t1.permute(dims=(1, 0, 2))
	print('t2->', t2)
	print('t2形状->', t2.shape)
	# t1形状修改为 (5, 3, 4)
	t3 = t1.permute(dims=(2, 0, 1))
	print('t3->', t3)
	print('t3形状->', t3.shape)


# tensor.view(shape=): 修改连续张量的形状, 操作等同于reshape()
# tensor.is_contiugous(): 判断张量是否连续, 返回True/False  张量经过transpose/permute处理变成不连续
# tensor.contiugous(): 将张量转为连续张量
def dm04():
	torch.manual_seed(0)
	t1 = torch.randint(low=0, high=10, size=(3, 4))
	print('t1->', t1)
	print('t1形状->', t1.shape)
	print('t1是否连续->', t1.is_contiguous())
	# 修改张量形状
	t2 = t1.view((4, 3))
	print('t2->', t2)
	print('t2形状->', t2.shape)
	print('t2是否连续->', t2.is_contiguous())
	# 张量经过transpose操作
	t3 = t1.transpose(dim0=1, dim1=0)
	print('t3->', t3)
	print('t3形状->', t3.shape)
	print('t3是否连续->', t3.is_contiguous())
	# 修改张量形状
	# view
	# contiugous(): 转换成连续张量
	t4 = t3.contiguous().view((3, 4))
	print('t4->', t4)
	t5 = t3.reshape(shape=(3, 4))
	print('t5->', t5)
	print('t5是否连续->', t5.is_contiguous())


if __name__ == '__main__':
	# dm01()
	# dm02()
	# dm03()
	dm04()

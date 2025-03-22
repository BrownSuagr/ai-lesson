import torch


# 运算: 张量和数值之间运算, 张量和张量之间运算
# + - * / -
# add(other=) sub() mul() div() neg()  不修改原张量
# add_() sub_() mul_() div_() neg_()  修改原张量

def dm01():
	# 创建张量
	t1 = torch.tensor(data=[1, 2, 3, 4])
	# 张量和数值运算
	t2 = t1 + 10
	print('t2->', t2)
	# 张量之间运算, 对应位置的元素进行计算
	t3 = t1 + t2
	print('t3->', t3)

	# add() 不修改原张量
	t1.add(other=100)
	t4 = torch.add(input=t1, other=100)
	print('t4->', t4)

	# neg_() 修改原张量, 负号
	t5 = t1.neg_()
	print('t1->', t1)
	print('t5->', t5)


if __name__ == '__main__':
	dm01()

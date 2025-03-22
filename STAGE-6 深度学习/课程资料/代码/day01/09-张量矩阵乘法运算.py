import torch


# 矩阵乘法: (n, m) * (m, p) = (n, p)  第一个矩阵的行和第二个矩阵的列相乘  @  torch.matmul(input=, ohter=)
def dm01():
	# (2, 2)
	t1 = torch.tensor(data=[[1, 2],
							[3, 4]])
	# (2, 3)
	t2 = torch.tensor(data=[[5, 6, 7],
							[8, 9, 10]])

	# @
	t3 = t1 @ t2
	print('t3->', t3)
	# torch.matmul(): 不同形状, 只要后边维度符合矩阵乘法规则即可
	t4 = torch.matmul(input=t1, other=t2)
	print('t4->', t4)


if __name__ == '__main__':
	dm01()

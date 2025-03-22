import torch


def dm01():
	# 创建张量
	t1 = torch.tensor(data=[[1., 2, 3, 4],
							[5, 6, 7, 8]])

	# dim=0 按列
	# dim=1 按行
	# 平均值
	print('所有值平均值->', t1.mean())
	print('按列平均值->', t1.mean(dim=0))
	print('按行平均值->', t1.mean(dim=1))
	# 求和
	print('所有值求和->', t1.sum())
	print('按列求和->', t1.sum(dim=0))
	print('按行求和->', t1.sum(dim=1))
	# sqrt: 开方 平方根
	print('所有值开方->', t1.sqrt())
	# pow: 幂次方  x^n
	# exponent:几次方
	print('幂次方->', torch.pow(input=t1, exponent=2))
	# exp: 指数 e^x  张量的元素值就是x
	print('指数->', torch.exp(input=t1))
	# log: 对数  log(x)->以e为底  log2()  log10()
	print('以e为底对数->', torch.log(input=t1))
	print('以2为底对数->', t1.log2())
	print('以10为底对数->', t1.log10())


if __name__ == '__main__':
	dm01()

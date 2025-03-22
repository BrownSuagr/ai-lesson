import torch
import torch.nn as nn


# dropout随机失活: 每批次样本训练时,随机让一部分神经元死亡,防止一些特征对结果影响大(防止过拟合)
def dm01():
	# todo:1-创建隐藏层输出结果
	# float(): 转换成浮点类型张量
	t1 = torch.randint(low=0, high=10, size=(1, 4)).float()
	print('t1->', t1)
	# todo:2-进行下一层加权求和计算
	linear1 = nn.Linear(in_features=4, out_features=4)
	l1 = linear1(t1)
	print('l1->', l1)
	# todo:3-进行激活值计算
	output = torch.sigmoid(l1)
	print('output->', output)
	# todo:4-对激活值进行dropout处理  训练阶段
	# p: 失活概率
	dropout = nn.Dropout(p=0.4)
	d1 = dropout(output)
	print('d1->', d1)


if __name__ == '__main__':
	dm01()
